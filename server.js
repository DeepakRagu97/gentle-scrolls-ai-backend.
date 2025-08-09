// Gentle Scrolls — Unified AI Backend for Story/Poem/Recipe Generators
// Single-file Express server. Deploy on Replit/Render/Railway/Fly.
//
// ENV REQUIRED:
//   OPENAI_API_KEY
// OPTIONAL:
//   PORT
//   CORS_ORIGINS                e.g. https://gentlescrolls.com,https://www.gentlescrolls.com
//   RATE_LIMIT_RPM              default 60
//   MAX_WORDS                   default 500 (free cap)
//   MAX_WORDS_PREMIUM           default 2000
//   STRIPE_SECRET_KEY           sk_test_...
//   STRIPE_PRICE_ID             price_...
//   SITE_URL                    https://gentlescrolls.com
//   TOKEN_SECRET                random long string

import express from "express";
import rateLimit from "express-rate-limit";
import fetch from "node-fetch";
import crypto from "crypto";
import Stripe from "stripe";

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "1mb" }));

/* ---------------------- CORS ---------------------- */
const origins = (
  process.env.CORS_ORIGINS ||
  "https://gentlescrolls.com,https://www.gentlescrolls.com,https://*.gentlescrolls.com"
)
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

app.use((req, res, next) => {
  const origin = req.headers.origin;
  const ok =
    !origin ||
    origins.some((o) => {
      if (o.startsWith("https://*.")) {
        const root = o.replace("https://*.", "");
        return origin === `https://${root}` || origin.endsWith(`.${root}`);
      }
      return origin === o;
    });

  if (ok && origin) res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  // allow our premium header as well:
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, X-Premium-Token",
  );

  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

/* ---------------------- Rate limit ---------------------- */
app.use(
  "/api/",
  rateLimit({
    windowMs: 60_000,
    max: Number(process.env.RATE_LIMIT_RPM || 60),
    standardHeaders: true,
    legacyHeaders: false,
  }),
);

/* ---------------------- Health ---------------------- */
app.get("/health", (req, res) => {
  res.json({ ok: true, ts: Date.now(), model: "gpt-4o-mini" });
});

/* ---------------------- OpenAI ---------------------- */
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) console.warn("⚠️  Missing OPENAI_API_KEY");

function sysPrompt(kind) {
  if (kind === "story")
    return "You are a vivid, imaginative short-story writer. Write clean, engaging prose with a beginning, middle and end.";
  if (kind === "poem")
    return "You are a helpful poetry model. Deliver evocative, original poems with clear line breaks.";
  if (kind === "recipe")
    return "You are a friendly chef. Output clear sections: Title, Servings, Ingredients (metric + imperial), Steps, Tips.";
  return "You are a helpful writing assistant.";
}

async function callOpenAI({ system, user, maxTokens }) {
  const r = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      input: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      max_output_tokens: maxTokens, // <-- was 1400, now dynamic
      temperature: 0.9,
    }),
  });

  if (!r.ok) {
    const t = await r.text();
    throw new Error(`OpenAI error: ${t}`);
  }
  const data = await r.json();
  const text =
    data.output_text ||
    data.output?.[0]?.content?.[0]?.text ||
    data.choices?.[0]?.message?.content ||
    "";
  return text.trim();
}

/* ---------------------- Premium tokens ---------------------- */
const TOKEN_SECRET =
  process.env.TOKEN_SECRET || crypto.randomBytes(32).toString("hex");

function sign(payload) {
  const data = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const sig = crypto
    .createHmac("sha256", TOKEN_SECRET)
    .update(data)
    .digest("base64url");
  return `${data}.${sig}`;
}
function verify(token) {
  if (!token || !token.includes(".")) return null;
  const [data, sig] = token.split(".");
  const good = crypto
    .createHmac("sha256", TOKEN_SECRET)
    .update(data)
    .digest("base64url");
  if (sig !== good) return null;
  try {
    return JSON.parse(Buffer.from(data, "base64url").toString());
  } catch {
    return null;
  }
}
function isPremium(req) {
  const token = req.headers["x-premium-token"];
  const data = verify(token);
  return !!(data && data.t === "premium" && data.exp > Date.now());
}

/* ---------------------- Word limits ---------------------- */
const MAX_WORDS = Number(process.env.MAX_WORDS || 500); // free cap
const MAX_WORDS_PREMIUM = Number(process.env.MAX_WORDS_PREMIUM || 2000);

const clampWords = (n, premium = false) => {
  const cap = premium ? MAX_WORDS_PREMIUM : MAX_WORDS;
  return Math.min(Math.max(Number(n) || 300, 50), cap);
};

function userPrompt(kind, payload, premium = false) {
  const {
    prompt = "",
    genre = "General",
    tone = "Neutral",
    audience = "General",
    words = 500,
    style = "Default",
    dietary = "None",
    time = "Any",
  } = payload || {};
  const w = clampWords(words, premium);

  if (kind === "story") {
    const min = Math.round(w * 0.9);
    const max = Math.round(w * 1.1);
    return `Write a short story for a ${audience} audience in a ${tone} ${genre} style based on this premise: ${prompt}.
  Target length: ${w} words (between ${min} and ${max} words). 
  Finish with a satisfying conclusion. Do not stop mid-sentence.`;
  }

  if (kind === "poem")
    return `Write a ${w}-word poem in ${style} style for the following brief: ${prompt}. Audience: ${audience}. Tone: ${tone}.`;
  if (kind === "recipe")
    return `Create a recipe (~${w} words) based on: ${prompt}. Consider dietary preference: ${dietary}. Max time: ${time}. Output: title, servings, ingredients (metric + imperial), steps, tips.`;
  return `Task: ${prompt}`;
}

/* ---------------------- Generation handler ---------------------- */
async function handleGenerate(req, res, kind) {
  try {
    const premium = isPremium(req);

    // pull requested words for budgeting (default 500)
    const requested = Number(req.body?.words) || 500;
    const targetWords = clampWords(requested, premium);

    // ~1.6 tokens/word + 200 buffer; cap higher for premium
    const maxTokens = Math.min(
      Math.round(targetWords * 1.6) + 200,
      premium ? 6000 : 1800,
    );

    const system = sysPrompt(kind);
    const user = userPrompt(kind, req.body || {}, premium);
    const content = await callOpenAI({ system, user, maxTokens });

    res.set("Cache-Control", "private, max-age=30");
    res.json({ kind, content, premium, targetWords, maxTokens });
  } catch (err) {
    console.error(err);
    res
      .status(500)
      .json({
        error: "Generation failed",
        details: String(err?.message || err),
      });
  }
}

/* ---------------------- Stripe ---------------------- */
const STRIPE_KEY = process.env.STRIPE_SECRET_KEY || "";
const STRIPE_PRICE = process.env.STRIPE_PRICE_ID || "";
const SITE_URL = process.env.SITE_URL || "https://gentlescrolls.com";
const stripe = STRIPE_KEY ? new Stripe(STRIPE_KEY) : null;

// Start Stripe Checkout (monthly subscription)
app.post("/api/premium/checkout", async (req, res) => {
  try {
    if (!stripe || !STRIPE_PRICE)
      return res.status(500).json({ error: "Stripe not configured" });
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      line_items: [{ price: STRIPE_PRICE, quantity: 1 }],
      success_url: `${SITE_URL}/unlock?sid={CHECKOUT_SESSION_ID}`,
      cancel_url: `${SITE_URL}/ai-short-story-generator?canceled=1`,
      allow_promotion_codes: true,
    });
    res.json({ url: session.url });
  } catch (e) {
    console.error(e);
    res.status(500).json({
      error: "Stripe session failed",
      details: String(e.message || e),
    });
  }
});

// Verify session and mint premium token
app.get("/api/premium/verify", async (req, res) => {
  try {
    if (!stripe)
      return res.status(500).json({ error: "Stripe not configured" });
    const sessionId = req.query.session_id || req.query.sid;

    if (!sessionId)
      return res.status(400).json({ error: "Missing session_id" });

    const session = await stripe.checkout.sessions.retrieve(sessionId, {
      expand: ["subscription"],
    });
    if (session.status !== "complete" || session.payment_status !== "paid") {
      return res.status(402).json({ error: "Payment not completed" });
    }

    const sub = session.subscription;
    const active =
      sub && (sub.status === "active" || sub.status === "trialing");
    if (!active)
      return res.status(403).json({ error: "No active subscription" });

    const exp = Date.now() + 1000 * 60 * 60 * 24 * 30; // 30 days
    const token = sign({
      t: "premium",
      exp,
      sid: session.id,
      sub: typeof sub === "string" ? sub : sub.id,
    });
    res.json({ premium: true, token, expires: exp });
  } catch (e) {
    console.error(e);
    res
      .status(500)
      .json({ error: "Verify failed", details: String(e.message || e) });
  }
});

/* ---------------------- Routes ---------------------- */
app.post("/api/story", (req, res) => handleGenerate(req, res, "story"));
app.post("/api/poem", (req, res) => handleGenerate(req, res, "poem"));
app.post("/api/recipe", (req, res) => handleGenerate(req, res, "recipe"));

app.post("/api/generate", (req, res) => {
  const type = (req.body?.type || "").toLowerCase();
  if (!["story", "poem", "recipe"].includes(type)) {
    return res
      .status(400)
      .json({ error: "Invalid type. Use story|poem|recipe" });
  }
  return handleGenerate(req, res, type);
});

app.get("/api/meta", (req, res) => {
  res.json({
    model: "gpt-4o-mini",
    maxWords: MAX_WORDS,
    maxWordsPremium: MAX_WORDS_PREMIUM,
    rateLimitRpm: Number(process.env.RATE_LIMIT_RPM || 60),
  });
});

/* ---------------------- 404 ---------------------- */
app.use((req, res) => res.status(404).json({ error: "Not found" }));

/* ---------------------- Listen ---------------------- */
const PORT = process.env.PORT || 8080;
app.listen(PORT, () =>
  console.log(`✅ Gentle Scrolls AI Backend running on :${PORT}`),
);

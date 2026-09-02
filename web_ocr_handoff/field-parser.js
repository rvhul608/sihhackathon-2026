const FIELD_NAMES = [
  "mrp", "net_quantity", "manufacturing_date", "expiry_date", "manufacturer_name",
  "manufacturer_address", "consumer_care_contact", "unit_sale_price", "country_of_origin",
];

const absent = () => ({ value: null, confidence: 0, present: false });
const setIfFound = (result, field, value, confidence, minConfidence) => {
  if (value && confidence >= minConfidence && !result[field].present) result[field] = { value, confidence: Math.round(Math.max(0, Math.min(1, confidence)) * 100) / 100, present: true };
};

const matchMoney = (text, label) => {
  const match = text.match(new RegExp(`${label}\\s*(?:\\(?\\s*(?:incl\\.?\\s*of\\s*all\\s*taxes?)?\\s*\\)?\\s*)?[:\\-]?\\s*((?:₹|rs\\.?|inr)\\s*\\d+(?:[,.]\\d{1,2})?)`, "i"));
  return match ? match[1].replace(/\s+/g, "") : null;
};
const matchDate = (text, label) => {
  const match = text.match(new RegExp(`${label}\\s*[:\\-]?\\s*((?:\\d{1,2}[/-])?\\d{1,2}[/-]\\d{2,4}|\\d{4}[/-]\\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\s*[-/]?\\s*\\d{2,4})`, "i"));
  return match?.[1].trim() ?? null;
};
const looksLikeAddress = (text) => /\d|(?:road|rd\.?|street|st\.?|nagar|district|india|pin)\b/i.test(text);

/** Exact JS port of vision_pipeline/parsing.py. Lines are { text, confidence }. */
export function parseDeclarations(lines, { minConfidence = 0 } = {}) {
  const result = Object.fromEntries(FIELD_NAMES.map((name) => [name, absent()]));
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const normalized = line.text.trim().replace(/\s+/g, " ");
    const confidence = Number(line.confidence);
    const lowered = normalized.toLowerCase();
    setIfFound(result, "mrp", matchMoney(normalized, "(?:m\\.?(?:r\\.?)?p\\.?|maximum retail price)"), confidence, minConfidence);
    const quantity = normalized.match(/(?:net\s*(?:qty|quantity|wt|weight|vol(?:ume)?)?)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:kg|g|mg|l|ml|litre?s?|pcs?|pieces?|nos?))/i);
    setIfFound(result, "net_quantity", quantity ? quantity[1].replace(/\s+/g, "") : null, confidence, minConfidence);
    setIfFound(result, "manufacturing_date", matchDate(normalized, "(?:mfg|mfd|manufactur(?:ed|ing)|packed)(?:\\s*date)?"), confidence, minConfidence);
    setIfFound(result, "expiry_date", matchDate(normalized, "(?:exp(?:iry)?|use\\s*by|best\\s*before)(?:\\s*date)?"), confidence, minConfidence);
    setIfFound(result, "unit_sale_price", matchMoney(normalized, "(?:unit\\s*(?:sale\\s*)?price|price\\s*per\\s*(?:kg|g|l|ml))"), confidence, minConfidence);
    if (/(?:consumer|customer)\s*(?:care|service|helpline)|toll\s*free|contact/i.test(normalized)) {
      const contact = normalized.match(/(?:\+?91[-\s]?)?(?:0?\d{3,5}[-\s]?)?\d{5,8}|\d{3,5}[-\s]\d{4,8}/);
      setIfFound(result, "consumer_care_contact", contact?.[0].trim(), confidence, minConfidence);
    }
    const origin = normalized.match(/(?:country\s*of\s*origin|made\s*in)\s*[:\-]?\s*([A-Za-z ]{3,})/i);
    setIfFound(result, "country_of_origin", origin?.[1].trim().replace(/[ .]+$/, ""), confidence, minConfidence);
    const manufacturer = /\b(?:manufactured\s+by|manufactured\s*&\s*marketed\s+by|marketed\s+by|packed\s+by)\b/i;
    if (manufacturer.test(lowered)) {
      const value = normalized.split(manufacturer).pop().replace(/^\s*[:\-]?\s*/, "").trim();
      setIfFound(result, "manufacturer_name", value, confidence, minConfidence);
      const next = lines[index + 1];
      if (next && looksLikeAddress(next.text)) setIfFound(result, "manufacturer_address", next.text.trim(), Number(next.confidence), minConfidence);
    } else if (/\b(?:address|add\.)\b/i.test(lowered)) {
      const address = normalized.split(/\b(?:address|add\.)\b\s*[:\-]?/i).pop().trim();
      setIfFound(result, "manufacturer_address", address, confidence, minConfidence);
    }
  }
  return result;
}

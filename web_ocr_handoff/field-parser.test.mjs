import assert from "node:assert/strict";
import { parseDeclarations } from "./field-parser.js";

const fields = parseDeclarations([
  { text: "Net Qty: 500 g", confidence: 0.94 }, { text: "MRP: ₹120.00 (incl. of all taxes)", confidence: 0.98 },
  { text: "MFD: 03/2026", confidence: 0.91 }, { text: "Best Before: 12/2026", confidence: 0.89 },
  { text: "Manufactured by: ABC Foods Pvt Ltd", confidence: 0.96 }, { text: "12 Industrial Road, Kochi, Kerala 682001", confidence: 0.92 },
  { text: "Consumer Care: 1800-123-4567", confidence: 0.95 }, { text: "Country of Origin: India", confidence: 0.97 },
]);
assert.equal(fields.mrp.value, "₹120.00"); assert.equal(fields.net_quantity.value, "500g");
assert.equal(fields.manufacturing_date.value, "03/2026"); assert.equal(fields.expiry_date.value, "12/2026");
assert.equal(fields.manufacturer_name.value, "ABC Foods Pvt Ltd"); assert.equal(fields.manufacturer_address.present, true);
assert.equal(fields.country_of_origin.value, "India");
assert.deepEqual(parseDeclarations([{ text: "MRP: Rs. 99", confidence: .88 }]).expiry_date, { value: null, confidence: 0, present: false });
assert.deepEqual(
  parseDeclarations([{ text: "MRP: Rs. 99", confidence: .49 }], { minConfidence: .5 }).mrp,
  { value: null, confidence: 0, present: false },
);
console.log("field-parser parity fixtures passed");

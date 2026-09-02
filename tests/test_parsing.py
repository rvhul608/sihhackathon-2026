import unittest

from vision_pipeline.parsing import ParsedLine, parse_declarations


class DeclarationParsingTests(unittest.TestCase):
    def test_extracts_common_clean_label_declarations(self):
        fields = parse_declarations([
            ParsedLine("Net Qty: 500 g", 0.94),
            ParsedLine("MRP: ₹120.00 (incl. of all taxes)", 0.98),
            ParsedLine("MFD: 03/2026", 0.91),
            ParsedLine("Best Before: 12/2026", 0.89),
            ParsedLine("Manufactured by: ABC Foods Pvt Ltd", 0.96),
            ParsedLine("12 Industrial Road, Kochi, Kerala 682001", 0.92),
            ParsedLine("Consumer Care: 1800-123-4567", 0.95),
            ParsedLine("Country of Origin: India", 0.97),
        ])
        self.assertEqual(fields["mrp"]["value"], "₹120.00")
        self.assertEqual(fields["net_quantity"]["value"], "500g")
        self.assertEqual(fields["manufacturing_date"]["value"], "03/2026")
        self.assertEqual(fields["expiry_date"]["value"], "12/2026")
        self.assertEqual(fields["manufacturer_name"]["value"], "ABC Foods Pvt Ltd")
        self.assertTrue(fields["manufacturer_address"]["present"])
        self.assertEqual(fields["country_of_origin"]["value"], "India")

    def test_keeps_absent_field_distinct(self):
        fields = parse_declarations([ParsedLine("MRP: Rs. 99", 0.88)])
        self.assertFalse(fields["expiry_date"]["present"])
        self.assertIsNone(fields["expiry_date"]["value"])
        self.assertEqual(fields["expiry_date"]["confidence"], 0.0)

    def test_server_threshold_does_not_publish_low_confidence_declarations(self):
        fields = parse_declarations(
            [ParsedLine("MRP: Rs. 99", 0.49)], min_confidence=0.50
        )
        self.assertEqual(
            fields["mrp"], {"value": None, "confidence": 0.0, "present": False}
        )


if __name__ == "__main__":
    unittest.main()

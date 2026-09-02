import unittest
from unittest.mock import Mock, patch

from vision_pipeline.extractor import (
    _apply_glare_penalty,
    _run_paddle_ocr,
    _validate_request,
)
from vision_pipeline.parsing import ParsedLine


class ServerPipelineTests(unittest.TestCase):
    def test_glare_below_warning_does_not_change_confidence(self):
        lines = [ParsedLine("MRP: Rs. 99", 0.9)]
        self.assertEqual(_apply_glare_penalty(lines, 0.025), lines)

    def test_glare_reduces_confidence(self):
        penalized = _apply_glare_penalty([ParsedLine("MRP: Rs. 99", 0.9)], 0.10)
        self.assertLess(penalized[0].confidence, 0.5)

    def test_invalid_category_and_threshold_are_rejected_before_ocr(self):
        with self.assertRaisesRegex(ValueError, "product_category"):
            _validate_request("medicine", 0.5)
        with self.assertRaisesRegex(ValueError, "field_confidence_threshold"):
            _validate_request("other", 1.01)

    def test_ocr_discards_empty_lines_and_keeps_line_confidence(self):
        ocr = Mock()
        ocr.ocr.return_value = [[
            [None, ("  MRP: Rs. 99  ", 0.91)],
            [None, ("   ", 0.99)],
            None,
        ]]
        with patch("vision_pipeline.extractor._paddle_ocr", return_value=ocr):
            self.assertEqual(
                _run_paddle_ocr(object()), [ParsedLine("MRP: Rs. 99", 0.91)]
            )


if __name__ == "__main__":
    unittest.main()

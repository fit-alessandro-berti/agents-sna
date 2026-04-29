from __future__ import annotations

import base64
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents_sna.benchmark_runner import (
    IMAGE_PROMPT,
    answer_file_path,
    clean_model_name,
    list_question_paths,
    load_question_prompt,
    safe_question_slug,
)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_clean_model_name_matches_benchmark_style(self) -> None:
        self.assertEqual(
            clean_model_name("openai/gpt-5.4-mini:verification heavy"),
            "openaigpt-5.4-miniverification_heavy",
        )

    def test_answer_file_path_uses_txt_for_png_questions(self) -> None:
        answer_path = answer_file_path(
            answers_dir=Path("answers"),
            run_slug="run",
            question_path=Path("cat07_01_ocdfg.png"),
        )

        self.assertEqual(answer_path, Path("answers/run_cat07_01_ocdfg.txt"))

    def test_list_question_paths_filters_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions_dir = Path(directory)
            (questions_dir / "b.txt").write_text("b", encoding="utf-8")
            (questions_dir / "a.png").write_bytes(b"png")
            (questions_dir / "__init__.py").write_text("", encoding="utf-8")

            paths = list_question_paths(questions_dir, include_images=True)

        self.assertEqual([path.name for path in paths], ["a.png", "b.txt"])

    def test_load_text_question_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.txt"
            path.write_text("Question text", encoding="utf-8")

            prompt = load_question_prompt(path)

        self.assertEqual(prompt, "Question text")

    def test_load_png_question_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.png"
            path.write_bytes(b"image-bytes")

            prompt = load_question_prompt(path)

        expected_image = base64.b64encode(b"image-bytes").decode("ascii")
        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[0]["text"], IMAGE_PROMPT)
        self.assertEqual(
            prompt[1]["image_url"]["url"],
            f"data:image/png;base64,{expected_image}",
        )

    def test_safe_question_slug(self) -> None:
        self.assertEqual(safe_question_slug(Path("cat 01?/x.txt")), "x")


if __name__ == "__main__":
    unittest.main()

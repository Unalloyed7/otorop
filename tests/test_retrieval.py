from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from documents import DocumentLoadError, load_documents, split_document
from retrieval import SearchIndex, build_index, evaluate_applicability

ROOT = Path(__file__).resolve().parents[1]
KB_ZIP = ROOT / "data" / "knowledge-base.zip"
KB_DIR = ROOT / "data" / "knowledge-base"


@unittest.skipUnless(KB_ZIP.is_file(), "Локальная база организаторов не установлена")
class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_index(KB_ZIP)

    def test_loads_all_21_documents_and_no_demo_files(self) -> None:
        self.assertEqual(len(self.index.documents), 21)
        self.assertTrue(all("demo" not in doc["source_file"].casefold() for doc in self.index.documents.values()))
        self.assertTrue(all(doc["source_file"].endswith(".txt") for doc in self.index.documents.values()))

    @unittest.skipUnless(KB_DIR.is_dir(), "Распакованная база организаторов не установлена")
    def test_ids_are_stable_for_zip_and_directory(self) -> None:
        from_zip = {doc["source_file"]: doc["document_id"] for doc in load_documents(KB_ZIP)}
        from_directory = {doc["source_file"]: doc["document_id"] for doc in load_documents(KB_DIR)}
        self.assertEqual(from_zip, from_directory)
        self.assertEqual(self.index.fingerprint, build_index(KB_DIR).fingerprint)
        self.assertNotEqual(
            self.index.fingerprint,
            SearchIndex(load_documents(KB_DIR), max_chunk_chars=1200).fingerprint,
        )

    def test_every_citation_matches_source_coordinates(self) -> None:
        for evidence in self.index.chunks:
            document = self.index.get_document(evidence["document_id"])
            json.dumps(evidence, ensure_ascii=False)
            self.assertEqual(
                document["text"][evidence["start_char"]:evidence["end_char"]],
                evidence["text"],
            )

    def test_finds_clause_far_beyond_starter_snippet(self) -> None:
        results = self.index.search(
            "арендатор возвращает помещение по акту в состоянии нормального износа",
            filters={"contract_number": "А-08/2024"},
        )
        self.assertTrue(results)
        self.assertGreater(results[0]["start_char"], 1200)
        self.assertEqual(results[0]["section_reference"], "6.3")

    def test_distinguishes_similar_counterparties(self) -> None:
        trade = self.index.search("гарантийный срок Альфа-Трейд", top_k=5)
        self.assertEqual(trade[0]["contract_number"], "П-17/2024")
        trading = self.index.search("гарантийный срок Альфа-Трейдинг", top_k=5)
        self.assertEqual(trading[0]["contract_number"], "ПТ-05/2023")
        filtered = self.index.search(
            "гарантийный срок",
            filters={"counterparty": "Альфа-Трейдинг"},
            top_k=10,
        )
        self.assertTrue(filtered)
        self.assertTrue(all(item["counterparty"] == "Альфа-Трейдинг" for item in filtered))
        ambiguous = self.index.search("гарантийный срок Альфа", top_k=10)
        self.assertTrue(
            {"Альфа-Трейд", "Альфа-Трейдинг"}
            <= {item["counterparty"] for item in ambiguous if item["counterparty"]}
        )

    def test_as_of_selects_historical_policy(self) -> None:
        results = self.index.search(
            "срок согласования юридическим отделом",
            filters={"document_type": "internal_policy", "as_of": "2024-08-01"},
            top_k=10,
        )
        self.assertTrue(results)
        self.assertTrue(all("2023" in item["source_file"] for item in results))
        self.assertNotIn("2025", " ".join(item["source_file"] for item in results))

    def test_explicit_document_dates_keep_distinct_meanings(self) -> None:
        p17 = next(
            doc for doc in self.index.documents.values()
            if doc["contract_number"] == "П-17/2024" and doc["document_type"] == "agreement"
        )
        policy_2025 = next(
            doc for doc in self.index.documents.values()
            if "ред 2025" in doc["source_file"]
        )
        self.assertEqual(p17["signed_date"], "2024-02-12")
        self.assertEqual(p17["effective_date"], "2024-02-12")
        self.assertEqual(p17["valid_to"], "2027-01-01")
        self.assertEqual(policy_2025["approval_date"], "2025-02-17")
        self.assertEqual(policy_2025["edition"], "2025-03-01")
        self.assertEqual(policy_2025["valid_from"], "2025-03-01")

    def test_contract_search_returns_agreement_and_amendment(self) -> None:
        results = self.index.search(
            "адрес доставки П-17/2024",
            filters={"contract_number": "П-17/2024"},
            top_k=10,
        )
        self.assertEqual(
            {item["document_type"] for item in results},
            {"agreement", "amendment"},
        )
        self.assertTrue(any(
            item["document_type"] == "agreement" and item["section_reference"] == "3.2"
            for item in results
        ))
        self.assertTrue(any(
            item["document_type"] == "amendment" and item["section_reference"] == "3.2"
            for item in results
        ))
        amendment = next(item for item in results if item["document_type"] == "amendment")
        self.assertIn("П-17/2024", amendment["source_contract_number"])

    def test_compliance_letter_links_to_current_policy(self) -> None:
        letter = next(
            doc for doc in self.index.documents.values()
            if "Письмо комплаенс-службы К-12" in doc["source_file"]
        )
        policy = next(
            doc for doc in self.index.documents.values()
            if "Регламент согласования договоров ред 2025" in doc["source_file"]
        )
        self.assertIn(policy["document_id"], letter["linked_document_ids"])
        self.assertIn(letter["document_id"], policy["linked_document_ids"])
        historical_policy = next(
            doc for doc in self.index.documents.values()
            if "Регламент согласования договоров ред 2023" in doc["source_file"]
        )
        self.assertNotIn(historical_policy["document_id"], letter["linked_document_ids"])

    def test_amendment_boundary_is_strict(self) -> None:
        amendment = next(
            doc for doc in self.index.documents.values()
            if doc["document_type"] == "amendment"
        )
        evidence = next(
            chunk for chunk in self.index.chunks
            if chunk["document_id"] == amendment["document_id"]
            and chunk["applicability_conditions"]
        )
        self.assertEqual(evaluate_applicability(evidence, "2024-10-01")["status"], "not_applicable")
        self.assertEqual(evaluate_applicability(evidence, "2024-10-02")["status"], "applicable")
        self.assertEqual(evaluate_applicability(evidence)["status"], "unknown")
        filtered = self.index.search(
            "адрес доставки П-17/2024",
            filters={
                "contract_number": "П-17/2024",
                "specification_signed_at": "2024-10-01",
                "only_applicable": True,
            },
        )
        self.assertFalse(any(item["document_type"] == "amendment" for item in filtered))

    def test_archive_rule_remains_searchable_without_promoting_injection(self) -> None:
        results = self.index.search("срок хранения оригинала договора в архиве", top_k=5)
        self.assertTrue(results)
        self.assertTrue(any("5 (пять) лет" in item["text"] for item in results))
        self.assertTrue(any("90 календарных дней" in item["text"] for item in self.index.chunks))
        self.assertTrue(all(
            item["text"] == self.index.get_document(item["document_id"])["text"][item["start_char"]:item["end_char"]]
            for item in results
        ))

    def test_payment_query_retrieves_contract_clause_despite_document_injection(self) -> None:
        results = self.index.search("срок оплаты по договору П-17/2024", top_k=10)
        self.assertTrue(any(
            item["document_type"] == "agreement"
            and item["contract_number"] == "П-17/2024"
            and item["section_reference"] == "4.2"
            and "30 (тридцати) календарных дней" in item["text"]
            for item in results
        ))
        self.assertTrue(any(
            "90 календарных дней" in item["text"]
            for item in self.index.chunks
        ))

    def test_missing_article_is_never_synthesized(self) -> None:
        results = self.index.search("статья 67 Трудового кодекса", top_k=10)
        self.assertTrue(all(item["text"] in self.index.get_document(item["document_id"])["text"] for item in results))
        self.assertFalse(any(
            item["section_reference"] == "67" and item["document_type"] == "law"
            for item in results
        ))

    @unittest.skipUnless(
        (ROOT / "data" / "demo_questions.csv").is_file(),
        "Открытые демовопросы организаторов не установлены",
    )
    def test_demo_questions_retrieve_expected_provisions_in_top_10(self) -> None:
        expected = {
            "1": [("Договор аренды А-08-2024", "3.1"), ("Договор аренды А-08-2024", "3.2")],
            "2": [("ТК_ст70", "Статья 70")],
            "4": [("Договор аренды А-08-2024", "3.1")],
            "5": [("Регламент согласования договоров ред 2023", "2.4")],
            "6": [("Договор поставки П-17-2024", "6.1")],
            "7": [("Договор аренды А-08-2024", "4.3")],
            "8": [("Договор поставки П-17-2024", "6.2")],
            "9": [("Договор поставки ПТ-05-2023", "4.1"), ("Договор поставки ПТ-05-2023", "4.2")],
            "10": [("Регламент согласования договоров ред 2025", "1.3")],
            "11": [("Служебная записка об архиве", None)],
            "12": [("ТК_ст70", "Статья 70")],
            "13": [("152ФЗ_ст9", "Статья 9")],
            "14": [("ТК_ст59", "Статья 59")],
            "15": [("Договор аренды А-08-2024", "6.3")],
            "16": [("Договор поставки П-17-2024", "2.1")],
            "17": [("Договор поставки ПТ-05-2023", "6.1")],
            "18": [("Регламент согласования договоров ред 2025", "3.2")],
        }
        questions_path = ROOT / "data" / "demo_questions.csv"
        with questions_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = {row["id"]: row["question"] for row in csv.DictReader(stream)}
        self.assertEqual(set(rows), {str(number) for number in range(1, 19)})

        for question_id, required in expected.items():
            filters = {"as_of": "2024-08-01"} if question_id == "5" else None
            results = self.index.search(rows[question_id], filters, top_k=10)
            for source_part, reference in required:
                self.assertTrue(
                    any(
                        source_part in result["source_file"]
                        and (reference is None or result["section_reference"] == reference)
                        and (
                            reference is not None
                            or "не более 3" in result["text"]
                        )
                        for result in results
                    ),
                    f"Вопрос {question_id}: не найдено положение {source_part}, {reference!r}",
                )
        maternity = self.index.search(rows["3"], top_k=10)
        self.assertFalse(any(
            "берем" in result["text"].casefold() or "родам" in result["text"].casefold()
            for result in maternity
        ))

    def test_neighbor_and_document_lookup(self) -> None:
        result = self.index.search("гарантийный срок П-17/2024", top_k=1)[0]
        self.assertEqual(self.index.get_chunk(result["chunk_id"])["chunk_id"], result["chunk_id"])
        neighbors = self.index.get_neighbors(result["chunk_id"])
        self.assertTrue(any(item["chunk_id"] == result["chunk_id"] for item in neighbors))

    def test_changed_corpus_changes_fingerprint_and_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            source = next(KB_DIR.glob("*Договор поставки П-17-2024*.txt"))
            copied = target / source.name
            shutil.copyfile(source, copied)
            before = build_index(target)
            copied.write_text(copied.read_text(encoding="utf-8") + "\nТестовое изменение.\n", encoding="utf-8")
            after = build_index(target)
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertNotEqual(
            next(iter(before.documents)),
            next(iter(after.documents)),
        )

    def test_corrupt_text_document_has_explicit_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.txt"
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(DocumentLoadError, "broken.txt"):
                load_documents(path.parent)

    def test_utf8_bom_russian_name_and_crlf_are_preserved(self) -> None:
        expected = "Заголовок\r\n1.1. Текст положения.\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Положение.txt"
            path.write_bytes(b"\xef\xbb\xbf" + expected.encode("utf-8"))
            document = load_documents(path.parent)[0]
            chunk = split_document(document)[1]
        self.assertEqual(document["text"], expected)
        self.assertEqual(
            document["text"][chunk["start_char"]:chunk["end_char"]],
            chunk["text"],
        )


if __name__ == "__main__":
    unittest.main()

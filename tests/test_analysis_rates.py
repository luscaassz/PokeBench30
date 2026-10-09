import csv
import tempfile
import unittest
from pathlib import Path

from pokebench30.evaluation.analysis import build_agent_summary


class AgentSummaryRateTests(unittest.TestCase):
    def test_voluntary_and_forced_switch_rates_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            with (run_dir / "battles.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "generation", "agent_1", "agent_2",
                        "result_agent_1", "turns"
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "generation": 9,
                    "agent_1": "A",
                    "agent_2": "B",
                    "result_agent_1": "win",
                    "turns": 10,
                })

            with (run_dir / "actions.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "generation", "agent", "action_type",
                        "voluntary_switch", "forced_switch"
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "generation": 9, "agent": "A", "action_type": "move",
                    "voluntary_switch": False, "forced_switch": False
                })
                writer.writerow({
                    "generation": 9, "agent": "A", "action_type": "switch",
                    "voluntary_switch": True, "forced_switch": False
                })
                writer.writerow({
                    "generation": 9, "agent": "A", "action_type": "switch",
                    "voluntary_switch": False, "forced_switch": True
                })

            rows = build_agent_summary(run_dir)
            a = next(r for r in rows if r["agent"] == "A")
            self.assertAlmostEqual(a["switch_rate"], 2 / 3, places=6)
            self.assertAlmostEqual(a["voluntary_switch_rate"], 1 / 3, places=6)
            self.assertAlmostEqual(a["forced_switch_rate"], 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()

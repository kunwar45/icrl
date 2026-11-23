# ABOUTME: Tests for the LeetCode anchor task used by GRAFT: the prompt hides the tests, and the reward is 1 only when the returned solution passes every hidden test in the sandbox
# ABOUTME: Run: pytest tests/test_leetcode_anchor_adapter.py -q
import json

from src.environments.leetcode_anchor.leetcode_anchor_adapter import chat_messages, score_replies

ITEM = {
    "id": "two-sum",
    "difficulty": "Medium",
    "problem": "Given an array of integers nums and an integer target, return indices of the two numbers that add up to target.",
    "starter_code": "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        ",
    "entry_point": "Solution().twoSum",
    "test": "def check(candidate):\n    assert candidate(nums = [3, 3], target = 6) == [0, 1]\n"
    "    assert candidate(nums = [2, 7, 11, 15], target = 9) == [0, 1]\n",
}
GOOD = (
    "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
    "        seen = defaultdict(int)\n"
    "        for i, n in enumerate(nums):\n"
    "            if target - n in seen:\n                return [seen[target - n], i]\n"
    "            seen[n] = i\n"
)


def reply(code, key="solution.py"):
    return f"<think>reasoning</think>\n{json.dumps({key: code})}"


def test_prompt_shows_the_problem_and_starter_code_but_not_the_tests():
    msgs = chat_messages(ITEM)
    text = msgs[1]["content"]
    assert msgs[0]["role"] == "system" and ITEM["problem"] in text
    assert json.dumps({"solution.py": ITEM["starter_code"]}) in text
    assert "candidate(" not in text and "test.py" not in text


def test_reward_needs_every_hidden_test_to_pass():
    wrong = GOOD.replace("return [seen[target - n], i]", "return [0, 0]")
    hang = "class Solution:\n    def twoSum(self, nums, target):\n        while True:\n            pass\n"
    # a reply cannot earn reward by returning its own test file: there is no test file to return
    own_test = f'<think>x</think>\n{json.dumps({"solution.py": wrong, "test.py": "def check(candidate):\n    pass\n"})}'
    replies = [reply(GOOD), reply(wrong), reply("def nothing(): pass"), reply(GOOD, key="other.py"), "no json", own_test]
    scores = score_replies([ITEM] * len(replies), replies)
    assert [s["reward"] for s in scores] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [s["parsed"] for s in scores] == [True, True, True, False, False, True]
    (s,) = score_replies([ITEM], [reply(hang)])
    assert s["reward"] == 0.0

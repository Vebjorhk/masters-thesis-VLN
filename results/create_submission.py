import json

with open("r2r_test_submission_non-panoramic_full_inst_0.json", "r") as f:
    data = json.load(f)


with open("r2r_test_submission_non-panoramic_full_inst_1.json", "r") as f:
    data2 = json.load(f)


with open("r2r_test_submission_non-panoramic_full_inst_2.json", "r") as f:
    data3 = json.load(f)


data.extend(data2)
data.extend(data3)

with open("test_submission.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
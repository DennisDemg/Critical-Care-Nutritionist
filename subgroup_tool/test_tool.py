from subgroup_tool import run


def test_subgroup_tool_minimal():

    payload = {
        "Age": 57,
        "CCI": 1,
        "Hemoglobin": 7.5,
        "Albumin": 2.794,
        "Creatinine": 2.3391,
        "BUN": 70.4778,
        "BCR": 30.13030653,
        "PF_ratio": 350,
        "Urine_output": 1800,
        "SOFA": 9
    }

    result = run(payload)

    print("\n Subgroup Prediction Result:")
    print(result)

    assert result["ok"] is True


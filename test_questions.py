test_cases = [
    {
        "question": "What should happen if a server overheats?",
        "expected_topic": "Server Overheating Procedure",
    },
    {
        "question": "What should happen during a power outage?",
        "expected_topic": "Power Outage Procedure",
    },
    {
        "question": "What are visitor access requirements?",
        "expected_topic": "Visitor Access Procedure",
    },
    {
        "question": "How many vacation days do employee receive?",
        "expected_topic": None,
    },
]

answer_test_cases = [
    {
        "question": "What should happen if a server overheats?",
        "expected_keywords": [
            "monitoring dashboard",
            "operations lead",
            "airflow",
        ],
    },
    {
        "question": "What should happen during a power outage?",
        "expected_keywords": [
            "backup generators",
            "UPS",
            "incident-management",
        ],
    },
    {
        "question": "What are the visitor access requirements",
        "expected_keywords": [
            "identification",
            "authorized employee",
        ],
    },
]
"""
Quick Answers Module

File Responsibility:
    Provides fast, pre-defined answers for common energy topics
    without requiring LLM calls.

Inputs:
    - Topic keyword or question

Outputs:
    - Pre-written answer string

Assumptions:
    - Topics are energy-related
    - Answers are accurate and up-to-date

Failure Modes:
    - Unknown topics return default message
"""


# Common quick answers dictionary
QUICK_ANSWERS = {
    'ac': (
        "Set AC to 24-26°C to save up to 6% per degree. "
        "Use ceiling fans to distribute cool air. "
        "Clean filters monthly for 5-15% efficiency gain."
    ),
    'led': (
        "LED bulbs use 80% less electricity than incandescent and last 25x longer. "
        "Switching all lights to LED can save ₹2,000-5,000/year."
    ),
    'solar': (
        "A 3kW solar system generates 12-15 units/day, saving ₹3,000-4,500/month. "
        "PM Surya Ghar offers up to ₹78,000 subsidy."
    ),
    'carbon': (
        "India's grid emission factor is 0.82 kg CO₂/kWh. "
        "Reducing 100 units/month saves 82 kg CO₂, equivalent to 4 trees."
    ),
    'off-peak': (
        "Running appliances during off-peak hours (11 PM - 6 AM) "
        "can reduce bills by 15-25% with time-of-use tariffs."
    ),
    'standby': (
        "Phantom loads from standby devices account for 5-10% of energy use. "
        "Use power strips with switches to eliminate standby waste."
    ),
    '5-star': (
        "5-star appliances use 20-30% less energy than 3-star. "
        "Higher upfront cost is recovered in 2-3 years through savings."
    ),
    'inverter': (
        "Inverter ACs/refrigerators save 30-50% energy by adjusting compressor speed. "
        "BLDC fans use 60% less than regular fans."
    ),
    'refrigerator': (
        "Keep fridge at 3-4°C and freezer at -18°C. "
        "Don't place hot food inside. 5-star fridges use 30-40% less energy."
    ),
    'washing': (
        "Wash with full loads in cold water when possible. "
        "Running during off-peak hours saves money on time-of-use tariffs."
    )
}

DEFAULT_ANSWER = (
    "I don't have a quick answer for that topic. "
    "Please ask a detailed question for a comprehensive response."
)


def get_quick_answer(topic: str) -> str:
    """
    Get a quick answer for common energy topics.
    
    Purpose: Fast response without LLM for common questions.
    
    Inputs:
        topic: Topic keyword or phrase
        
    Outputs:
        Pre-written answer string
        
    Side effects: None
    """
    topic_lower = topic.lower()
    
    for key, answer in QUICK_ANSWERS.items():
        if key in topic_lower:
            return answer
    
    return DEFAULT_ANSWER


def get_all_topics() -> list:
    """Get list of available quick answer topics."""
    return list(QUICK_ANSWERS.keys())

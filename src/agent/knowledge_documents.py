"""
Knowledge Documents Module

File Responsibility:
    Contains energy-related knowledge documents for the RAG system.
    These documents provide context for the AI energy advisor.

Inputs:
    None (data module)

Outputs:
    List of knowledge documents with id, category, title, content

Assumptions:
    - Documents are focused on Indian energy context
    - Content is factually accurate and up-to-date

Failure Modes:
    None (stateless data)
"""

from typing import List, Dict


ENERGY_KNOWLEDGE_BASE: List[Dict] = [
    # Energy Saving Tips
    {
        "id": "tip_1",
        "category": "energy_saving",
        "title": "AC Temperature Optimization",
        "content": """Setting your air conditioner to 24-26°C instead of lower temperatures can save up to 6% energy for each degree raised. Using ceiling fans along with AC helps distribute cool air more efficiently, allowing you to set a higher temperature while maintaining comfort. Regular cleaning of AC filters every month improves efficiency by 5-15%."""
    },
    {
        "id": "tip_2", 
        "category": "energy_saving",
        "title": "LED Lighting Benefits",
        "content": """LED bulbs consume 80% less electricity than incandescent bulbs and 50% less than CFLs. They last 25 times longer than incandescent bulbs. Replacing all home lighting with LEDs can save ₹2,000-5,000 annually on electricity bills. Motion sensors in low-traffic areas further reduce lighting costs."""
    },
    {
        "id": "tip_3",
        "category": "energy_saving", 
        "title": "Off-Peak Usage Strategy",
        "content": """Electricity rates are often lower during off-peak hours (typically 11 PM to 6 AM). Running heavy appliances like washing machines, dishwashers, and water heaters during these hours can reduce bills by 15-25%. Many utilities offer time-of-use tariffs that reward off-peak consumption."""
    },
    {
        "id": "tip_4",
        "category": "energy_saving",
        "title": "Standby Power Reduction",
        "content": """Phantom loads or standby power from electronics can account for 5-10% of home energy use. Using power strips with switches, unplugging chargers when not in use, and choosing appliances with low standby consumption can save ₹1,000-2,000 annually. Smart power strips automatically cut power to devices in standby mode."""
    },
    {
        "id": "tip_5",
        "category": "energy_saving",
        "title": "Refrigerator Efficiency",
        "content": """Refrigerators run 24/7 and account for 15-20% of home electricity use. Keep the thermostat at 3-4°C for the fridge and -18°C for the freezer. Don't place hot food directly inside. Ensure door seals are tight. Keep the refrigerator away from heat sources and leave space behind for ventilation. A 5-star rated refrigerator uses 30-40% less energy than lower-rated models."""
    },
    
    # Solar Energy
    {
        "id": "solar_1",
        "category": "solar",
        "title": "Rooftop Solar Benefits",
        "content": """A 3kW rooftop solar system can generate 12-15 units of electricity per day, saving ₹3,000-4,500 monthly on electricity bills. Solar panels have a lifespan of 25+ years with minimal maintenance. Net metering allows you to sell excess power back to the grid. Solar panels also provide roof insulation, reducing cooling costs."""
    },
    {
        "id": "solar_2",
        "category": "solar",
        "title": "PM Surya Ghar Yojana",
        "content": """Under the PM Surya Ghar Muft Bijli Yojana, households can get subsidies up to ₹78,000 for installing rooftop solar systems. The scheme covers 1-3 kW systems with Central Financial Assistance (CFA) of ₹30,000/kW for systems up to 2 kW and ₹18,000/kW for 2-3 kW capacity. Apply through the National Portal for Rooftop Solar. Many states offer additional incentives."""
    },
    {
        "id": "solar_3",
        "category": "solar",
        "title": "Solar System Sizing",
        "content": """To size a solar system, calculate your average daily consumption in kWh and divide by 4-5 (average peak sun hours in India). A typical household using 10 units/day needs a 2-3 kW system. Consider roof space (10 sq.m per kW), orientation (south-facing is optimal), and shading. Battery storage adds 30-50% to system cost but provides backup during outages."""
    },
    
    # Carbon Footprint
    {
        "id": "carbon_1",
        "category": "carbon",
        "title": "India Electricity Emission Factor",
        "content": """India's electricity grid has an emission factor of approximately 0.82 kg CO2 per kWh (as per Central Electricity Authority). This means every unit of electricity consumed generates 820 grams of CO2. The emission factor varies by source: coal (0.91 kg/kWh), natural gas (0.45 kg/kWh), solar (0.05 kg/kWh lifecycle), and wind (0.01 kg/kWh). Reducing consumption directly reduces your carbon footprint."""
    },
    {
        "id": "carbon_2",
        "category": "carbon",
        "title": "Carbon Offset with Trees",
        "content": """A mature tree absorbs approximately 22 kg of CO2 per year. An average Indian household consuming 200 units/month generates about 164 kg CO2 monthly or 2 tonnes annually. This would require planting 90 trees to offset. Reducing consumption is more effective than offsetting - cutting 50 units/month saves the equivalent of 22 trees annually."""
    },
    
    # Appliances
    {
        "id": "appliance_1",
        "category": "appliances",
        "title": "Energy Star Ratings",
        "content": """BEE (Bureau of Energy Efficiency) star ratings help identify efficient appliances. A 5-star AC uses 20-30% less energy than a 3-star model. For a 1.5-ton AC running 8 hours daily, this saves ₹3,000-5,000 annually. Always check the annual energy consumption (in kWh) on the label - lower is better. 5-star appliances cost more upfront but save money over their lifetime."""
    },
    {
        "id": "appliance_2",
        "category": "appliances",
        "title": "Inverter Technology",
        "content": """Inverter ACs and refrigerators adjust compressor speed based on load, unlike conventional fixed-speed units that cycle on/off. This provides 30-50% energy savings, faster cooling, quieter operation, and longer lifespan. The higher upfront cost is typically recovered within 2-3 years through energy savings. BLDC (Brushless DC) fans use 60% less electricity than regular fans."""
    },
    
    # Tariffs
    {
        "id": "tariff_1",
        "category": "tariffs",
        "title": "Electricity Tariff Slabs in India",
        "content": """Most Indian states have telescopic/slab-based tariffs where the rate increases with consumption. Typical domestic slabs: 0-100 units (₹3-4/unit), 101-200 units (₹4.5-5.5/unit), 201-300 units (₹6-7/unit), above 300 units (₹7.5-8.5/unit). Some states have Time-of-Day tariffs with lower rates during off-peak hours. Fixed charges and taxes add 20-30% to the energy charges."""
    },
    
    # Smart Home
    {
        "id": "smart_1",
        "category": "smart_home",
        "title": "Smart Home Energy Management",
        "content": """Smart plugs and energy monitors help track consumption of individual appliances. Smart thermostats learn your preferences and optimize heating/cooling schedules. Smart lighting with motion sensors and schedules reduces wastage. Home energy management systems (HEMS) can reduce overall consumption by 10-20% through automation and insights. Voice assistants can control devices remotely."""
    }
]


def get_all_documents() -> List[Dict]:
    """Return all knowledge documents."""
    return ENERGY_KNOWLEDGE_BASE.copy()


def get_documents_by_category(category: str) -> List[Dict]:
    """Get documents filtered by category."""
    return [doc for doc in ENERGY_KNOWLEDGE_BASE if doc['category'] == category]


def get_all_categories() -> List[str]:
    """Get unique category names."""
    return list(set(doc['category'] for doc in ENERGY_KNOWLEDGE_BASE))

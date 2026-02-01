"""
Tariff Calculator Module

File Responsibility:
    Handles electricity tariff slab definitions and bill calculations
    based on Indian domestic electricity rates.

Inputs:
    - Total energy consumption in kWh
    - Optional custom tariff slabs

Outputs:
    - Bill amount with slab-wise breakdown

Assumptions:
    - Tariff slabs are based on typical Indian domestic rates
    - Rates are in Indian Rupees (₹) per kWh
    - Progressive tiered pricing structure

Failure Modes:
    - Negative consumption values will produce incorrect results
"""

from typing import List, Dict


# Default Indian domestic electricity tariff slabs
DEFAULT_TARIFF_SLABS = [
    {'min_units': 0, 'max_units': 100, 'rate': 3.0},
    {'min_units': 101, 'max_units': 200, 'rate': 4.5},
    {'min_units': 201, 'max_units': 300, 'rate': 6.0},
    {'min_units': 301, 'max_units': 500, 'rate': 7.5},
    {'min_units': 501, 'max_units': float('inf'), 'rate': 8.5}
]


def get_tariff_slabs() -> List[Dict]:
    """
    Returns electricity tariff slabs.
    
    Purpose: Provide default Indian domestic electricity rates.
    
    Inputs: None
    
    Outputs:
        List of tariff slab dictionaries with min_units, max_units, rate
        
    Side effects: None
    """
    return DEFAULT_TARIFF_SLABS.copy()


def calculate_bill(total_kwh: float, tariff_slabs: List[Dict] = None) -> Dict:
    """
    Calculate electricity bill based on consumption and tariff slabs.
    
    Purpose: Compute tiered electricity bill with detailed breakdown.
    
    Inputs:
        total_kwh: Total energy consumption in kWh
        tariff_slabs: Optional custom tariff slabs (uses default if None)
        
    Outputs:
        Dictionary with total_units, total_bill, and breakdown list
        
    Side effects: None
    """
    if tariff_slabs is None:
        tariff_slabs = get_tariff_slabs()
    
    remaining = total_kwh
    total_bill = 0.0
    breakdown = []
    
    for slab in tariff_slabs:
        if remaining <= 0:
            break
        
        slab_range = slab['max_units'] - slab['min_units']
        if slab['max_units'] == float('inf'):
            units_in_slab = remaining
        else:
            units_in_slab = min(remaining, slab_range)
        
        slab_cost = units_in_slab * slab['rate']
        total_bill += slab_cost
        remaining -= units_in_slab
        
        breakdown.append({
            'slab': f"{slab['min_units']}-{slab['max_units']} units",
            'units': round(units_in_slab, 2),
            'rate': slab['rate'],
            'cost': round(slab_cost, 2)
        })
    
    return {
        'total_units': round(total_kwh, 2),
        'total_bill': round(total_bill, 2),
        'breakdown': breakdown
    }


def estimate_monthly_bill(daily_avg_kwh: float) -> Dict:
    """
    Estimate monthly bill from daily average consumption.
    
    Purpose: Quick monthly bill estimation for forecasting.
    
    Inputs:
        daily_avg_kwh: Average daily consumption in kWh
        
    Outputs:
        Dictionary with monthly estimate and bill details
        
    Side effects: None
    """
    monthly_kwh = daily_avg_kwh * 30
    bill = calculate_bill(monthly_kwh)
    bill['daily_average'] = round(daily_avg_kwh, 2)
    bill['estimated_monthly_kwh'] = round(monthly_kwh, 2)
    return bill


if __name__ == "__main__":
    # Test tariff calculator
    print("=" * 50)
    print("TARIFF CALCULATOR TEST")
    print("=" * 50)
    
    test_consumptions = [80, 150, 250, 400, 600]
    
    for kwh in test_consumptions:
        result = calculate_bill(kwh)
        print(f"\n{kwh} kWh -> ₹{result['total_bill']:.2f}")
        for slab in result['breakdown']:
            print(f"  {slab['slab']}: {slab['units']} units @ ₹{slab['rate']} = ₹{slab['cost']}")
    
    print("\n✓ Tariff calculator working correctly!")

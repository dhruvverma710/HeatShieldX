STRINGS = {
    'en': {
        'page_title': "Citizen Route Planner",
        'origin': "Origin",
        'destination': "Destination",
        'departure_time': "Departure Time",
        'tradeoff': "Trade-off: Faster <-> Cooler",
        'find_routes': "Find Routes",
        'faster': "Faster",
        'cooler': "Cooler",
        'route_cards': "Routes",
        'comparison_table': "Comparison",
        'safe_stops': "Safe stops along this route",
        'stop_type': "Type",
        'stop_name': "Name",
        'detour': "Detour",
        'verified': "Verified",
        'amenities': "Amenities",
        'rating': "Rating",
        'data_unavailable': "Data unavailable",
        'sponsored': "Sponsored",
        'shade_not_verified': "shade not verified",
        'lower_exposure': "Lower modelled heat exposure",
        'no_gain': "No lower-exposure route found within the detour limit.",
        'disclaimer': "Note: Modelled prioritisation values. Not medical advice."
    }
}

def get_string(key, lang='en'):
    return STRINGS.get(lang, {}).get(key, key)

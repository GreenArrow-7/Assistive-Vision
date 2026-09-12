"""Outdoor navigation: estimate distance/time, then hand off to Google Maps.

The user asks "how far and how long"; Maps owns turn-by-turn. Before the
handoff we geocode the destination (OpenStreetMap Nominatim) and estimate
walking distance/time from the user's GPS fix: straight line x1.3 for street
routing, 4.5 km/h. Deliberately hedged in speech ("approximately", "about").
Any failure (no fix, no geocode result, network) degrades to the plain handoff
rather than blocking navigation.
"""
import json
import math
import urllib.parse
import urllib.request

from pydantic import BaseModel, Field, model_validator

USER_AGENT = "AssistiveVision/1.8 (assistive navigation; student project)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
WALK_KMH = 4.5
ROAD_FACTOR = 1.3          # straight line -> typical street distance
GEOCODE_TIMEOUT_S = 6.0


class NavigationRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=200)
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_location(self):
        self.destination = self.destination.strip()
        if not self.destination:
            raise ValueError("Destination must not be blank")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be supplied together")
        return self


def geocode(query: str, latitude=None, longitude=None, timeout=GEOCODE_TIMEOUT_S):
    """(lat, lon) of the best Nominatim hit, biased toward the user, or None."""
    params = {"format": "json", "limit": 1, "q": query}
    if latitude is not None and longitude is not None:
        # prefer results near the user without EXCLUDING far ones (bounded=0)
        params["viewbox"] = f"{longitude - 0.5},{latitude + 0.5},{longitude + 0.5},{latitude - 0.5}"
        params["bounded"] = 0
    req = urllib.request.Request(NOMINATIM + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def describe(destination: str, distance_m: float, duration_s: float) -> str:
    if distance_m < 1000:
        dist = f"{max(50, int(round(distance_m / 50.0)) * 50)} metres"
    else:
        dist = f"{distance_m / 1000:.1f} kilometres"
    mins = max(1, int(round(duration_s / 60)))
    when = f"about {mins} minute{'s' if mins != 1 else ''} on foot"
    return (f"{destination} is approximately {dist} away, {when}. "
            "Opening Google Maps for walking directions.")


class MapsHandoffProvider:
    def __init__(self, geocoder=None):
        self.geocoder = geocoder     # None = pure handoff, no network (tests)

    def route(self, destination, latitude=None, longitude=None):
        params = dict(api=1, destination=destination, travelmode="walking")
        if latitude is not None and longitude is not None:
            params["origin"] = f"{latitude},{longitude}"
        out = {
            "provider": "google_maps_handoff", "destination": destination,
            "resolved": False, "distance_m": None, "duration_s": None,
            "maps_url": "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(params),
            "speech": "I could not estimate the distance. "
                      "Opening Google Maps for walking directions.",
        }
        if latitude is None or longitude is None or self.geocoder is None:
            return out
        try:
            hit = self.geocoder(destination, latitude, longitude)
        except Exception:
            hit = None
        if not hit:
            return out
        distance = haversine_m(latitude, longitude, hit[0], hit[1]) * ROAD_FACTOR
        duration = distance / (WALK_KMH * 1000 / 3600)
        out.update(resolved=True, distance_m=round(distance), duration_s=round(duration),
                   speech=describe(destination, distance, duration))
        return out

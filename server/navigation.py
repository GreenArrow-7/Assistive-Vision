"""Offline routing fallback: Maps resolves the destination after explicit handoff."""
from urllib.parse import urlencode
from pydantic import BaseModel, Field, model_validator

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

class MapsHandoffProvider:
    def route(self, destination, latitude=None, longitude=None):
        params = dict(api=1, destination=destination, travelmode="walking")
        if latitude is not None and longitude is not None:
            params["origin"] = f"{latitude},{longitude}"
        return {
            "provider": "google_maps_handoff", "destination": destination,
            "resolved": False, "distance_m": None, "duration_s": None,
            "maps_url": "https://www.google.com/maps/dir/?" + urlencode(params),
            "speech": "Route distance and travel time are unavailable locally. "
                      "Open Google Maps to resolve your destination and get directions.",
        }

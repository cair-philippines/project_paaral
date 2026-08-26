"use client";

import Map, { Marker, NavigationControl, Popup } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import type { School } from "@/types/school";
import SchoolPopupCard from "@/components/molecules/SchoolPopupCard";

const QUEZON_CITY_CENTER = { longitude: 121.0437, latitude: 14.676 };

const MARKER_COLOR: Record<string, string> = {
  public: "#0038a8",
  private_esc: "#16a34a",
  private_no_esc: "#f59e0b",
};

function markerKey(school: School): string {
  if (school.school_type === "public") return "public";
  return school.is_esc_participating ? "private_esc" : "private_no_esc";
}

interface SchoolMapProps {
  schools: School[];
  selectedSchoolId: string | null;
  onSelectSchool: (school: School | null) => void;
}

/** Mapbox view of the browse page's filtered results. Only schools with
 * coordinates render as pins — roughly 40% of the real Quezon City dataset,
 * per the 2026-08 BigQuery pull; the list/card views cover the rest. */
export default function SchoolMap({
  schools,
  selectedSchoolId,
  onSelectSchool,
}: SchoolMapProps) {
  const mapped = schools.filter(
    (school): school is School & { latitude: number; longitude: number } =>
      school.latitude !== null && school.longitude !== null
  );

  const selectedSchool = mapped.find(
    (school) => school.school_id === selectedSchoolId
  );

  return (
    <div className="relative h-full w-full">
      <Map
        mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
        initialViewState={{ ...QUEZON_CITY_CENTER, zoom: 12 }}
        mapStyle="mapbox://styles/mapbox/light-v11"
        style={{ width: "100%", height: "100%" }}
        onClick={() => onSelectSchool(null)}
      >
        <NavigationControl position="top-right" />
        {mapped.map((school) => (
          <Marker
            key={school.school_id}
            longitude={school.longitude}
            latitude={school.latitude}
            onClick={(e) => {
              e.originalEvent.stopPropagation();
              onSelectSchool(school);
            }}
          >
            <span
              className="block cursor-pointer rounded-full border-2 border-white shadow-md transition"
              style={{
                width: school.school_id === selectedSchoolId ? 16 : 10,
                height: school.school_id === selectedSchoolId ? 16 : 10,
                backgroundColor: MARKER_COLOR[markerKey(school)],
              }}
            />
          </Marker>
        ))}

        {selectedSchool && (
          <Popup
            longitude={selectedSchool.longitude}
            latitude={selectedSchool.latitude}
            onClose={() => onSelectSchool(null)}
            closeOnClick={false}
            maxWidth="300px"
            offset={14}
          >
            <SchoolPopupCard school={selectedSchool} />
          </Popup>
        )}
      </Map>
    </div>
  );
}

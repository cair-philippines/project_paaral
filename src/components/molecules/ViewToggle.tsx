import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import ToggleButton from "@mui/material/ToggleButton";
import { Map, List, LayoutGrid } from "lucide-react";

export type BrowseViewMode = "map" | "list" | "card";

interface ViewToggleProps {
  value: BrowseViewMode;
  onChange: (value: BrowseViewMode) => void;
}

/** Three-way map/list/card toggle over one shared result set, matching the
 * Chile vitrina benchmark's browse pattern. */
export default function ViewToggle({ value, onChange }: ViewToggleProps) {
  return (
    <ToggleButtonGroup
      size="small"
      exclusive
      value={value}
      onChange={(_, next) => next && onChange(next as BrowseViewMode)}
    >
      <ToggleButton value="map" aria-label="Map view">
        <Map size={16} />
      </ToggleButton>
      <ToggleButton value="list" aria-label="List view">
        <List size={16} />
      </ToggleButton>
      <ToggleButton value="card" aria-label="Card view">
        <LayoutGrid size={16} />
      </ToggleButton>
    </ToggleButtonGroup>
  );
}

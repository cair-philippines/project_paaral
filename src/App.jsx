import React, { useMemo, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Clock3,
  Info,
  Layers,
  MapPin,
  Minus,
  Plus,
  Search,
  SlidersHorizontal,
  Sparkles,
  Star,
  X,
} from "lucide-react";

// --- SYNTHETIC DATA ---
const schools = [
  { id: "SCH001", name: "St. Mary's Academy of Taguig", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Bagumbayan", postal_code: "1630", lat: 14.5176, lng: 121.0509, tuition: 45000, esc_subsidy: 13000, net_cost: 32000, slots_total: 40, slots_available: 12, distance_km: 3.2, commute_minutes: 15, esc_rating: 4, religious_affiliation: "Sectarian" },
  { id: "SCH002", name: "Bagumbayan National High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Bagumbayan", postal_code: "1630", lat: 14.5211, lng: 121.0576, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 360, slots_available: 22, distance_km: 1.4, commute_minutes: 6, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH003", name: "Senator Renato Cayetano Memorial Science and Technology High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Ususan", postal_code: "1639", lat: 14.5382, lng: 121.0675, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 280, slots_available: 9, distance_km: 4.8, commute_minutes: 22, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH004", name: "Pateros Catholic School", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pateros", barangay: "San Roque", postal_code: "1620", lat: 14.5455, lng: 121.0699, tuition: 52000, esc_subsidy: 13000, net_cost: 39000, slots_total: 45, slots_available: 18, distance_km: 5.7, commute_minutes: 28, esc_rating: 5, religious_affiliation: "Sectarian" },
  { id: "SCH005", name: "Fort Bonifacio Christian Academy", type: "private_esc", sector: "non_sectarian", region: "NCR", province: "Metro Manila", municipality: "Makati City", barangay: "Cembo", postal_code: "1214", lat: 14.5538, lng: 121.0436, tuition: 47000, esc_subsidy: 13000, net_cost: 34000, slots_total: 42, slots_available: 17, distance_km: 7.1, commute_minutes: 31, esc_rating: 4, religious_affiliation: "Non-Sectarian" },
  { id: "SCH006", name: "Pasig Grace Christian School", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pasig City", barangay: "Maybunga", postal_code: "1607", lat: 14.5752, lng: 121.0837, tuition: 42000, esc_subsidy: 13000, net_cost: 29000, slots_total: 50, slots_available: 15, distance_km: 10.8, commute_minutes: 42, esc_rating: 4, religious_affiliation: "Sectarian" },
  { id: "SCH007", name: "St. Paul College Pasig", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pasig City", barangay: "Ugong", postal_code: "1604", lat: 14.5845, lng: 121.0797, tuition: 118000, esc_subsidy: 0, net_cost: 118000, slots_total: 60, slots_available: 21, distance_km: 11.6, commute_minutes: 45, esc_rating: 0, religious_affiliation: "Sectarian" },
  { id: "SCH008", name: "Imus National High School", type: "public", sector: null, region: "Region IV-A", province: "Cavite", municipality: "Imus City", barangay: "Poblacion", postal_code: "4103", lat: 14.4297, lng: 120.9367, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 420, slots_available: 44, distance_km: 21.4, commute_minutes: 58, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH009", name: "St. Edward Integrated School", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Cavite", municipality: "Imus City", barangay: "Buhay na Tubig", postal_code: "4103", lat: 14.4144, lng: 120.9577, tuition: 62000, esc_subsidy: 11000, net_cost: 51000, slots_total: 55, slots_available: 27, distance_km: 24.6, commute_minutes: 62, esc_rating: 4, religious_affiliation: "Sectarian" },
  { id: "SCH010", name: "Dasmarinas Integrated High School", type: "public", sector: null, region: "Region IV-A", province: "Cavite", municipality: "Dasmarinas City", barangay: "Zone IV", postal_code: "4114", lat: 14.3294, lng: 120.9366, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 500, slots_available: 63, distance_km: 34.9, commute_minutes: 78, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH011", name: "Cavite Christian School", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Cavite", municipality: "Bacoor City", barangay: "Molino III", postal_code: "4102", lat: 14.4117, lng: 120.9742, tuition: 38000, esc_subsidy: 11000, net_cost: 27000, slots_total: 48, slots_available: 8, distance_km: 22.2, commute_minutes: 54, esc_rating: 3, religious_affiliation: "Sectarian" },
  { id: "SCH012", name: "Southville International School Cavite", type: "private_no_esc", sector: "non_sectarian", region: "Region IV-A", province: "Cavite", municipality: "Bacoor City", barangay: "Habitat", postal_code: "4102", lat: 14.4338, lng: 120.9643, tuition: 142000, esc_subsidy: 0, net_cost: 142000, slots_total: 35, slots_available: 19, distance_km: 20.5, commute_minutes: 49, esc_rating: 0, religious_affiliation: "Non-Sectarian" },
  { id: "SCH013", name: "San Pedro Relocation Center National High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "San Pedro City", barangay: "Landayan", postal_code: "4023", lat: 14.3588, lng: 121.0536, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 390, slots_available: 17, distance_km: 25.7, commute_minutes: 64, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH014", name: "Binan City Science and Technology High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "Binan City", barangay: "San Antonio", postal_code: "4024", lat: 14.3371, lng: 121.0804, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 260, slots_available: 12, distance_km: 30.6, commute_minutes: 72, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH015", name: "Colegio de San Juan de Letran Calamba", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Laguna", municipality: "Calamba City", barangay: "Bucal", postal_code: "4027", lat: 14.1981, lng: 121.1653, tuition: 59000, esc_subsidy: 11000, net_cost: 48000, slots_total: 70, slots_available: 31, distance_km: 53.8, commute_minutes: 96, esc_rating: 5, religious_affiliation: "Sectarian" },
  { id: "SCH016", name: "Laguna BelAir School", type: "private_no_esc", sector: "non_sectarian", region: "Region IV-A", province: "Laguna", municipality: "Santa Rosa City", barangay: "Don Jose", postal_code: "4026", lat: 14.2825, lng: 121.0894, tuition: 98000, esc_subsidy: 0, net_cost: 98000, slots_total: 45, slots_available: 23, distance_km: 39.8, commute_minutes: 83, esc_rating: 0, religious_affiliation: "Non-Sectarian" },
  { id: "SCH017", name: "Malolos Integrated School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Guinhawa", postal_code: "3000", lat: 14.8527, lng: 120.816, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 430, slots_available: 51, distance_km: 48.2, commute_minutes: 92, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH018", name: "Meycauayan National High School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Meycauayan City", barangay: "Calvario", postal_code: "3020", lat: 14.7368, lng: 120.9608, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 470, slots_available: 28, distance_km: 32.7, commute_minutes: 75, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH019", name: "St. Anne's Catholic School of Bulacan", type: "private_esc", sector: "sectarian", region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Catmon", postal_code: "3000", lat: 14.8471, lng: 120.8111, tuition: 41000, esc_subsidy: 9000, net_cost: 32000, slots_total: 52, slots_available: 16, distance_km: 49.1, commute_minutes: 95, esc_rating: 4, religious_affiliation: "Sectarian" },
  { id: "SCH020", name: "Bulacan Ecumenical School", type: "private_esc", sector: "non_sectarian", region: "Region III", province: "Bulacan", municipality: "Marilao", barangay: "Loma de Gato", postal_code: "3019", lat: 14.7571, lng: 120.9488, tuition: 36000, esc_subsidy: 9000, net_cost: 27000, slots_total: 40, slots_available: 4, distance_km: 35.4, commute_minutes: 81, esc_rating: 3, religious_affiliation: "Non-Sectarian" },
  { id: "SCH021", name: "Our Lady of Guadalupe School San Jose del Monte", type: "private_no_esc", sector: "sectarian", region: "Region III", province: "Bulacan", municipality: "San Jose del Monte City", barangay: "Tungkong Mangga", postal_code: "3023", lat: 14.8167, lng: 121.0754, tuition: 74000, esc_subsidy: 0, net_cost: 74000, slots_total: 44, slots_available: 11, distance_km: 43.9, commute_minutes: 98, esc_rating: 0, religious_affiliation: "Sectarian" },
  { id: "SCH022", name: "Angeles City National Trade School", type: "public", sector: null, region: "Region III", province: "Pampanga", municipality: "Angeles City", barangay: "Pulungbulu", postal_code: "2009", lat: 15.1456, lng: 120.5881, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 520, slots_available: 76, distance_km: 91.5, commute_minutes: 132, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH023", name: "San Fernando High School", type: "public", sector: null, region: "Region III", province: "Pampanga", municipality: "City of San Fernando", barangay: "Dolores", postal_code: "2000", lat: 15.0287, lng: 120.6893, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 490, slots_available: 33, distance_km: 78.8, commute_minutes: 118, esc_rating: 0, religious_affiliation: "Public" },
  { id: "SCH024", name: "Holy Angel Academy", type: "private_esc", sector: "sectarian", region: "Region III", province: "Pampanga", municipality: "Angeles City", barangay: "Sto. Rosario", postal_code: "2009", lat: 15.1344, lng: 120.5906, tuition: 48000, esc_subsidy: 9000, net_cost: 39000, slots_total: 65, slots_available: 26, distance_km: 90.7, commute_minutes: 130, esc_rating: 5, religious_affiliation: "Sectarian" },
  { id: "SCH025", name: "Pampanga Central Institute", type: "private_esc", sector: "non_sectarian", region: "Region III", province: "Pampanga", municipality: "Mexico", barangay: "San Antonio", postal_code: "2021", lat: 15.0701, lng: 120.7219, tuition: 35000, esc_subsidy: 9000, net_cost: 26000, slots_total: 48, slots_available: 14, distance_km: 82.2, commute_minutes: 124, esc_rating: 4, religious_affiliation: "Non-Sectarian" },
  { id: "SCH026", name: "Clarkfield Learning Center", type: "private_no_esc", sector: "non_sectarian", region: "Region III", province: "Pampanga", municipality: "Mabalacat City", barangay: "Dau", postal_code: "2010", lat: 15.1842, lng: 120.5939, tuition: 128000, esc_subsidy: 0, net_cost: 128000, slots_total: 38, slots_available: 20, distance_km: 95.8, commute_minutes: 140, esc_rating: 0, religious_affiliation: "Non-Sectarian" },
];

const typeMeta = {
  public: { label: "Public", badge: "bg-[#1a4b8c] text-white", dot: "#3b82f6" },
  private_esc: { label: "Private with ESC", badge: "bg-[#16a34a] text-white", dot: "#22c55e" },
  private_no_esc: { label: "Private no ESC", badge: "bg-[#f59e0b] text-white", dot: "#f59e0b" },
};

const regionOptions = ["NCR", "Region III", "Region IV-A"];
const provinceOptions = ["Metro Manila", "Cavite", "Laguna", "Bulacan", "Pampanga"];
const municipalityOptions = [
  "Taguig City", "Pateros", "Makati City", "Pasig City", "Imus City", "Bacoor City",
  "Dasmarinas City", "San Pedro City", "Binan City", "Santa Rosa City", "Calamba City",
  "Meycauayan City", "Malolos City", "Marilao", "San Jose del Monte City", "Angeles City",
  "City of San Fernando", "Mexico", "Mabalacat City",
];
const barangayOptions = [
  "Bagumbayan", "Ususan", "San Roque", "Cembo", "Maybunga", "Ugong", "Poblacion",
  "Buhay na Tubig", "Zone IV", "Molino III", "Habitat", "Landayan", "San Antonio",
  "Bucal", "Don Jose", "Guinhawa", "Calvario", "Catmon", "Loma de Gato", "Tungkong Mangga",
  "Pulungbulu", "Dolores", "Sto. Rosario", "Dau",
];

const pesos = (value) =>
  value === 0 ? "Free" : new Intl.NumberFormat("en-PH", { style: "currency", currency: "PHP", maximumFractionDigits: 0 }).format(value);

const pct = (available, total) => Math.round((available / total) * 100);

const schoolTypeMatches = (school, selectedTypes) => {
  if (!selectedTypes.length) return true;
  return selectedTypes.includes(school.type);
};

const commuteBucketMatches = (school, buckets) => {
  if (!buckets.length) return true;
  return buckets.some((bucket) => {
    if (bucket === "under5") return school.commute_minutes < 5;
    if (bucket === "15to30") return school.commute_minutes >= 15 && school.commute_minutes <= 30;
    if (bucket === "over30") return school.commute_minutes > 30;
    return true;
  });
};

const slotTone = (school) => {
  const ratio = pct(school.slots_available, school.slots_total);
  if (school.slots_available === 0 || ratio <= 5) return "bg-[#dc2626]";
  if (ratio <= 20) return "bg-[#f59e0b]";
  return "bg-[#16a34a]";
};

const accordionDefaults = { location: true, distance: true, commute: true, tuition: true, subsidy: true, type: true };

// --- MAP GEOMETRY UTILS (Carto Light Styling) ---
const MAP_BOUNDS = { minLng: 120.52, maxLng: 121.25, minLat: 14.15, maxLat: 15.22 };
const MAP_WIDTH = 750;
const MAP_HEIGHT = 1000;

const WATER_COLOR = "#d1dce5"; 
const LAND_COLOR = "#f0f3f4";  
const HIGHWAY_COLOR = "#ffffff";
const HIGHWAY_BORDER = "#e2e8f0";

const project = (lat, lng) => {
  return {
    x: ((lng - MAP_BOUNDS.minLng) / (MAP_BOUNDS.maxLng - MAP_BOUNDS.minLng)) * MAP_WIDTH,
    y: MAP_HEIGHT - ((lat - MAP_BOUNDS.minLat) / (MAP_BOUNDS.maxLat - MAP_BOUNDS.minLat)) * MAP_HEIGHT,
  };
};

const projectPoint = (school) => project(school.lat, school.lng);

function MapPolygon({ points, fill, stroke, strokeWidth = "2", opacity = "1" }) {
  const projectedPoints = points.map(([lat, lng]) => `${project(lat, lng).x},${project(lat, lng).y}`).join(" ");
  return <polygon points={projectedPoints} fill={fill} stroke={stroke} strokeWidth={strokeWidth} opacity={opacity} />;
}

function MapPath({ points, stroke, strokeWidth, fill = "none", dasharray = "none", opacity = "1" }) {
  const d = points.map(([lat, lng], i) => `${i === 0 ? "M" : "L"} ${project(lat, lng).x} ${project(lat, lng).y}`).join(" ");
  return <path d={d} fill={fill} stroke={stroke} strokeWidth={strokeWidth} strokeDasharray={dasharray} strokeLinecap="round" strokeLinejoin="round" opacity={opacity} />;
}

function MapLabel({ lat, lng, text, fontSize = "13", fontWeight = "bold", opacity = "0.6" }) {
  const coords = project(lat, lng);
  return <text x={coords.x} y={coords.y} fill="#64748b" fontSize={fontSize} fontWeight={fontWeight} fontFamily="sans-serif" opacity={opacity} textAnchor="middle" letterSpacing="0.05em">{text}</text>;
}

const MANILA_BAY_POINTS = [[14.2, 120.52], [14.25, 120.72], [14.3, 120.8], [14.38, 120.86], [14.45, 120.92], [14.48, 120.9], [14.45, 120.95], [14.48, 120.98], [14.54, 120.99], [14.6, 120.96], [14.68, 120.94], [14.76, 120.88], [14.8, 120.75], [14.86, 120.52]];
const LAGUNA_LAKE_POINTS = [[14.53, 121.1], [14.45, 121.05], [14.35, 121.06], [14.28, 121.12], [14.2, 121.18], [14.2, 121.25], [14.5, 121.25], [14.5, 121.18]];
const TAAL_LAKE_POINTS = [[14.08, 120.94], [14.05, 121.02], [14.02, 121.1], [13.94, 121.08], [13.92, 120.99], [13.98, 120.92]];

const EDSA_POINTS = [[14.54, 120.99], [14.54, 121.02], [14.56, 121.04], [14.58, 121.06], [14.62, 121.05], [14.65, 121.03], [14.66, 120.99]];
const SLEX_POINTS = [[14.54, 121.02], [14.48, 121.04], [14.42, 121.04], [14.32, 121.08], [14.25, 121.1], [14.2, 121.13]];
const NLEX_POINTS = [[14.66, 120.99], [14.7, 120.98], [14.75, 120.95], [14.82, 120.88], [14.95, 120.78], [15.05, 120.7], [15.18, 120.59]];

// --- UI COMPONENTS ---
function FilterSection({ title, id, open, onToggle, children }) {
  return (
    <div className="border-b border-[#e2e4e9] py-4 last:border-b-0">
      <button type="button" onClick={() => onToggle(id)} className="flex w-full items-center justify-between text-left focus:outline-none">
        <span className="text-sm font-semibold text-[#1a1d23]">{title}</span>
        {open ? <ChevronUp className="h-4 w-4 text-[#6b7280]" /> : <ChevronDown className="h-4 w-4 text-[#6b7280]" />}
      </button>
      {open && <div className="mt-3 space-y-3">{children}</div>}
    </div>
  );
}

function CheckboxRow({ checked, label, sublabel, onChange }) {
  return (
    <label className="flex cursor-pointer items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-[#f8f9fb]">
      <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md border ${checked ? "border-[#1a4b8c] bg-[#1a4b8c]" : "border-[#d1d5db] bg-white"} transition-colors`}>
        {checked && <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />}
      </span>
      {/* Hidden input to prevent browser scroll jumping to focus */}
      <input type="checkbox" checked={checked} onChange={onChange} className="hidden" />
      <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
        <span className="truncate text-sm text-[#1a1d23]">{label}</span>
        {sublabel && <span className="shrink-0 text-xs text-[#9ca3af]">{sublabel}</span>}
      </span>
    </label>
  );
}

function RangePair({ min, max, value, onChange, format, step = 1 }) {
  const updateMin = (next) => onChange([Math.min(Number(next), value[1]), value[1]]);
  const updateMax = (next) => onChange([value[0], Math.max(Number(next), value[0])]);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between rounded-lg border border-[#e2e4e9] bg-white px-3 py-2">
        <span className="font-['SF_Mono','Fira_Code','Consolas',monospace] text-xs font-semibold text-[#1a4b8c]">{format(value[0])}</span>
        <span className="text-xs text-[#9ca3af]">to</span>
        <span className="font-['SF_Mono','Fira_Code','Consolas',monospace] text-xs font-semibold text-[#1a4b8c]">{format(value[1])}</span>
      </div>
      <div className="space-y-2">
        <input type="range" min={min} max={max} step={step} value={value[0]} onChange={(e) => updateMin(e.target.value)} className="h-2 w-full appearance-none rounded-lg bg-[#e2e4e9] accent-[#1a4b8c]" />
        <input type="range" min={min} max={max} step={step} value={value[1]} onChange={(e) => updateMax(e.target.value)} className="h-2 w-full appearance-none rounded-lg bg-[#e2e4e9] accent-[#1a4b8c]" />
      </div>
    </div>
  );
}

function SelectField({ value, onChange, options, placeholder }) {
  return (
    <div className="relative min-w-0">
      <select value={value} onChange={(e) => onChange(e.target.value)} className="h-10 w-full appearance-none rounded-lg border border-[#e2e4e9] bg-white px-3 pr-9 text-sm text-[#1a1d23] outline-none transition focus:border-[#1a4b8c] focus:ring-2 focus:ring-[#1a4b8c]/10 truncate">
        <option value="">{placeholder}</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 top-3 h-4 w-4 text-[#6b7280]" />
    </div>
  );
}

function Stars({ rating }) {
  if (!rating) return <span className="text-xs text-[#9ca3af]">Not applicable</span>;
  return (
    <span className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => <Star key={star} className={`h-3.5 w-3.5 ${star <= rating ? "fill-[#d4a843] text-[#d4a843]" : "fill-[#e5e7eb] text-[#e5e7eb]"}`} />)}
    </span>
  );
}

function ResultCard({ school, selected, onSelect }) {
  // SYSTEM VALIDATION: Check if subsidy is active [cite: 466]
  const isEscParticipant = school.esc_subsidy > 0;

  return (
    <button 
      type="button" 
      onClick={() => onSelect(school)} 
      className={`w-full rounded-xl border bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 ${selected ? "border-[#1a4b8c] ring-4 ring-[#1a4b8c]/10" : "border-[#e2e4e9]"}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="line-clamp-2 text-sm font-semibold leading-snug text-[#1a1d23]">{school.name}</h3>
          <p className="mt-1 text-xs text-[#6b7280] truncate">{school.municipality}, {school.province}</p>
        </div>
        <span className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-semibold ${isEscParticipant ? "bg-[#16a34a] text-white" : "bg-[#f59e0b] text-white"}`}>
          {isEscParticipant ? "ESC" : "Private"}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Distance</p>
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">{school.distance_km} km</p>
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Net Cost</p>
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">{pesos(school.net_cost)}</p>
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Slots</p>
          {/* FIXED: Only show number if ESC is active, otherwise show '-' [cite: 551] */}
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">
            {isEscParticipant ? school.slots_available : "—"}
          </p>
        </div>
      </div>

      {/* FIXED: Only show progress bar for ESC participants [cite: 420] */}
      {isEscParticipant && (
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#f0f1f4]">
          <div className={`h-full rounded-full ${slotTone(school)}`} style={{ width: `${pct(school.slots_available, school.slots_total)}%` }} />
        </div>
      )}
    </button>
  );
}

function SchoolInfoCard({ school, onClose }) {
  if (!school) return null;
  
  // SYSTEM VALIDATION: A school is an ESC participant ONLY if it has an active subsidy
  const isEscParticipant = school.esc_subsidy > 0;
  
  const meta = typeMeta[school.type];
  const availablePct = pct(school.slots_available, school.slots_total);
  const point = projectPoint(school);
  
  const xShift = point.x > (MAP_WIDTH / 2) ? "max(-105%, -1 * calc(100vw - 460px))" : "18px";
  const yShift = point.y > (MAP_HEIGHT / 2) ? "max(-85%, -1 * calc(100vh - 120px))" : "-18px";
  
  const cardPosition = {
    left: `${(point.x / MAP_WIDTH) * 100}%`,
    top: `${(point.y / MAP_HEIGHT) * 100}%`,
    transform: `translate(${xShift}, ${yShift})`,
  };

  return (
    <div
      className="absolute z-30 hidden w-[340px] max-h-[85vh] overflow-y-auto custom-scrollbar rounded-[20px] border border-[#e2e4e9] bg-white/95 p-5 shadow-[0_18px_48px_rgba(26,29,35,0.16),0_4px_12px_rgba(0,0,0,0.08)] backdrop-blur md:block transition-all duration-300"
      style={cardPosition}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: isEscParticipant ? "#22c55e" : "#f59e0b" }} />
            <span className="text-xs font-semibold uppercase tracking-[0.08em] text-[#6b7280] truncate">
              {isEscParticipant ? "ESC Participating" : "Private No ESC"}
            </span>
          </div>
          <h3 className="mt-3 text-lg font-semibold leading-tight text-[#1a1d23]">{school.name}</h3>
        </div>
        <button type="button" onClick={onClose} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[#e2e4e9] text-[#6b7280] transition hover:bg-[#f8f9fb] focus:outline-none">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 border-y border-[#e2e4e9] py-4">
        <div className="flex items-center gap-2"><MapPin className="h-4 w-4 text-[#1a4b8c]" /><span className="text-sm text-[#1a1d23]">{school.distance_km} km from you</span></div>
        <div className="flex items-center gap-2"><Clock3 className="h-4 w-4 text-[#1a4b8c]" /><span className="text-sm text-[#1a1d23]">~{school.commute_minutes} min</span></div>
      </div>

      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between text-sm"><span className="text-[#6b7280]">Tuition Fee</span><span className="font-semibold text-[#1a1d23]">{pesos(school.tuition)}</span></div>
        <div className="flex items-center justify-between text-sm">
            <span className="text-[#6b7280]">ESC Subsidy</span>
            <span className={`font-semibold ${isEscParticipant ? "text-[#16a34a]" : "text-[#9ca3af]"}`}>
                {isEscParticipant ? `-${pesos(school.esc_subsidy)}` : "None Available"}
            </span>
        </div>
        <div className="mt-2 flex items-center justify-between border-t border-[#e2e4e9] pt-3">
            <span className="text-sm font-semibold text-[#1a1d23]">Net Cost</span>
            <span className="text-xl font-bold text-[#1a4b8c]">{pesos(school.net_cost)}/yr</span>
        </div>
      </div>

      {/* FIXED: Slots are only visible for validated ESC participants */}
      {isEscParticipant ? (
        <div className="mt-5 rounded-xl bg-[#f8f9fb] p-4 border border-[#16a34a]/10">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-[#1a1d23]">Available ESC Slots</span>
            <span className="font-mono text-sm font-semibold text-[#1a1d23]">{school.slots_available} of {school.slots_total}</span>
          </div>
          <div className="mt-3 flex items-center gap-3">
            <div className="h-3 flex-1 overflow-hidden rounded-full bg-white border border-[#e2e4e9]">
              <div className={`h-full rounded-full ${slotTone(school)}`} style={{ width: `${availablePct}%` }} />
            </div>
            <span className="w-10 shrink-0 text-right text-xs font-semibold text-[#6b7280]">{availablePct}%</span>
          </div>
        </div>
      ) : (
        <div className="mt-5 rounded-xl bg-slate-50 p-4 border border-dashed border-slate-200">
            <p className="text-xs text-center text-slate-500 italic">This institution does not currently offer ESC subsidized slots.</p>
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div><p className="text-xs uppercase tracking-[0.08em] text-[#9ca3af]">ESC Rating</p><div className="mt-1"><Stars rating={school.esc_rating} /></div></div>
        <div><p className="text-xs uppercase tracking-[0.08em] text-[#9ca3af]">Religious</p><p className="mt-1 font-semibold text-[#1a1d23]">{school.religious_affiliation}</p></div>
      </div>
    </div>
  );
}

function PhilippinesMap({ filteredSchools, selectedSchool, hoveredId, onHover, onSelect, comingSoon }) {
  return (
    <div className="relative h-full w-full overflow-hidden bg-[#e2e8f0]">
      <div className="absolute right-5 top-5 z-10 flex flex-col overflow-hidden rounded-xl border border-[#e2e4e9] bg-white shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
        <button className="flex h-10 w-10 items-center justify-center border-b border-[#e2e4e9] text-[#1a4b8c] hover:bg-[#f8f9fb] focus:outline-none" type="button"><Plus className="h-5 w-5" /></button>
        <button className="flex h-10 w-10 items-center justify-center text-[#1a4b8c] hover:bg-[#f8f9fb] focus:outline-none" type="button"><Minus className="h-5 w-5" /></button>
      </div>

      <svg viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`} preserveAspectRatio="xMidYMid slice" className="h-full w-full" role="img" aria-label="Greater Manila and Luzon locator map">
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#3b82f6" floodOpacity="0.6"/></filter>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#000000" floodOpacity="0.25"/></filter>
        </defs>
        
        {/* Carto Light Base */}
        <rect x="0" y="0" width={MAP_WIDTH} height={MAP_HEIGHT} fill={LAND_COLOR} />
        <MapPolygon points={MANILA_BAY_POINTS} fill={WATER_COLOR} stroke="none" />
        <MapPolygon points={LAGUNA_LAKE_POINTS} fill={WATER_COLOR} stroke="none" />
        <MapPolygon points={TAAL_LAKE_POINTS} fill={WATER_COLOR} stroke="none" opacity="0.65" />
        
        {/* Carto Style Highways */}
        <MapPath points={NLEX_POINTS} stroke={HIGHWAY_BORDER} strokeWidth="8" />
        <MapPath points={NLEX_POINTS} stroke={HIGHWAY_COLOR} strokeWidth="5" />
        <MapPath points={SLEX_POINTS} stroke={HIGHWAY_BORDER} strokeWidth="8" />
        <MapPath points={SLEX_POINTS} stroke={HIGHWAY_COLOR} strokeWidth="5" />
        <MapPath points={EDSA_POINTS} stroke={HIGHWAY_BORDER} strokeWidth="8" />
        <MapPath points={EDSA_POINTS} stroke={HIGHWAY_COLOR} strokeWidth="5" />

        {/* Labels Map */}
        <MapLabel lat={15.06} lng={120.7} text="PAMPANGA" fontSize="18" fontWeight="800" opacity="0.3" />
        <MapLabel lat={14.86} lng={121.02} text="BULACAN" fontSize="18" fontWeight="800" opacity="0.3" />
        <MapLabel lat={14.62} lng={121.15} text="RIZAL" fontSize="18" fontWeight="800" opacity="0.3" />
        <MapLabel lat={14.32} lng={120.9} text="CAVITE" fontSize="18" fontWeight="800" opacity="0.3" />
        <MapLabel lat={14.24} lng={121.05} text="LAGUNA" fontSize="18" fontWeight="800" opacity="0.3" />
        <MapLabel lat={14.58} lng={121.02} text="NCR" fontSize="18" fontWeight="800" opacity="0.36" />

        <MapLabel lat={15.15} lng={120.59} text="Angeles" />
        <MapLabel lat={15.03} lng={120.69} text="San Fernando" />
        <MapLabel lat={14.84} lng={120.81} text="Malolos" />
        <MapLabel lat={14.73} lng={120.96} text="Meycauayan" />
        <MapLabel lat={14.62} lng={120.98} text="Manila" />
        <MapLabel lat={14.65} lng={121.05} text="Quezon City" />
        <MapLabel lat={14.52} lng={121.08} text="Taguig" />
        <MapLabel lat={14.42} lng={121.04} text="Muntinlupa" />
        <MapLabel lat={14.33} lng={120.94} text="Dasmarinas" />
        <MapLabel lat={14.46} lng={120.96} text="Bacoor" />
        <MapLabel lat={14.31} lng={121.11} text="Santa Rosa" />
        
        <g transform={`translate(${project(14.54, 120.78).x}, ${project(14.54, 120.78).y}) rotate(-45)`}><text fill="#94a3b8" fontSize="16" fontWeight="bold" letterSpacing="0.2em" opacity="0.6">MANILA BAY</text></g>
        <g transform={`translate(${project(14.36, 121.16).x}, ${project(14.36, 121.16).y}) rotate(15)`}><text fill="#94a3b8" fontSize="16" fontWeight="bold" letterSpacing="0.2em" opacity="0.6">LAGUNA DE BAY</text></g>

        {filteredSchools.map((school) => {
          const { x, y } = projectPoint(school);
          const isSelected = selectedSchool?.id === school.id;
          const isHovered = hoveredId === school.id;
          const scale = isSelected || isHovered ? 1.35 : 1;
          
          return (
            <g
              key={school.id}
              transform={`translate(${x} ${y}) scale(${scale})`}
              className="cursor-pointer transition-transform duration-300"
              onMouseEnter={() => onHover(school.id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(school)}
            >
              <circle r="12" fill="transparent" />
              {(isSelected || isHovered) && <circle r="9" fill={typeMeta[school.type].dot} opacity="0.4" className="animate-ping" />}
              
              <circle 
                r={isSelected ? "7" : isHovered ? "6" : "5"} 
                fill={typeMeta[school.type].dot} 
                stroke="#ffffff" 
                strokeWidth={isSelected ? "2.5" : "1.5"} 
                filter={isSelected ? "url(#glow)" : "url(#shadow)"}
                style={{ transition: 'all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)' }} 
              />

              {isHovered && !isSelected && (
                <g transform="translate(10 -20)">
                  <rect x="0" y="0" rx="6" ry="6" width={school.name.length * 6 + 16} height="24" fill="#ffffff" filter="url(#shadow)" />
                  <text x="8" y="16" fill="#1e293b" fontSize="11" fontFamily="sans-serif" fontWeight="bold">{school.name}</text>
                </g>
              )}
            </g>
          );
        })}
      </svg>

      <div className="absolute bottom-5 right-5 z-10 w-56 rounded-xl border border-[#e2e4e9] bg-white/95 p-4 shadow-[0_1px_3px_rgba(0,0,0,0.06)] backdrop-blur">
        <div className="mb-3 flex items-center gap-2">
          <Layers className="h-4 w-4 text-[#1a4b8c]" />
          <span className="text-sm font-semibold text-[#1a1d23]">School Options</span>
        </div>
        <div className="space-y-2">
          {Object.entries(typeMeta).map(([type, meta]) => (
            <div className="flex items-center justify-between text-xs" key={type}>
              <span className="flex items-center gap-2 text-[#6b7280]">
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: meta.dot }} />
                {meta.label}
              </span>
              <span className="font-semibold text-[#1a1d23]">
                {filteredSchools.filter((school) => school.type === type).length}
              </span>
            </div>
          ))}
        </div>
      </div>

      <SchoolInfoCard school={selectedSchool} onClose={() => onSelect(null)} />

      {comingSoon && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-[#f8f9fb]/82 p-8 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-[20px] border border-[#e2e4e9] bg-white p-7 text-center shadow-[0_18px_48px_rgba(26,29,35,0.16),0_4px_12px_rgba(0,0,0,0.08)]">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-[#1a4b8c]/10">
              <Sparkles className="h-6 w-6 text-[#1a4b8c]" />
            </div>
            <h2 className="mt-4 text-xl font-semibold text-[#1a1d23]">{comingSoon} is coming soon</h2>
            <p className="mt-2 text-sm leading-6 text-[#6b7280]">
              This live demo focuses on the family-facing Student View. The same access and capacity data model can power
              school operations and DepEd policy simulations.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PAARALStudentMockup() {
  const [activeView, setActiveView] = useState("Student View");
  const [searchTerm, setSearchTerm] = useState("");
  const [postalCode, setPostalCode] = useState("");
  const [region, setRegion] = useState("");
  const [province, setProvince] = useState("");
  const [municipality, setMunicipality] = useState("");
  const [barangay, setBarangay] = useState("");
  const [distance, setDistance] = useState([0, 100]);
  const [tuition, setTuition] = useState([0, 250000]);
  const [commuteBuckets, setCommuteBuckets] = useState([]);
  const [subsidies, setSubsidies] = useState([]);
  const [schoolTypes, setSchoolTypes] = useState([]);
  const [religious, setReligious] = useState([]);
  const [openSections, setOpenSections] = useState(accordionDefaults);
  const [selectedSchool, setSelectedSchool] = useState(schools[0]);
  const [hoveredId, setHoveredId] = useState(null);

  const filteredSchools = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    return schools.filter((school) => {
      const queryMatch =
        !query ||
        [school.name, school.type, school.region, school.province, school.municipality, school.barangay, school.religious_affiliation]
          .join(" ").toLowerCase().includes(query);
      const postalMatch = !postalCode || `${school.postal_code} ${school.municipality} ${school.barangay}`.toLowerCase().includes(postalCode.toLowerCase());
      const locationMatch = (!region || school.region === region) && (!province || school.province === province) && (!municipality || school.municipality === municipality) && (!barangay || school.barangay === barangay);
      const distanceMatch = school.distance_km >= distance[0] && school.distance_km <= distance[1];
      const tuitionMatch = school.tuition >= tuition[0] && school.tuition <= tuition[1];
      const subsidyMatch = !subsidies.length || subsidies.includes(school.esc_subsidy);
      const religiousMatch = !religious.length || religious.includes(school.sector) || (religious.includes("public") && school.type === "public");

      return queryMatch && postalMatch && locationMatch && distanceMatch && tuitionMatch && commuteBucketMatches(school, commuteBuckets) && subsidyMatch && schoolTypeMatches(school, schoolTypes) && religiousMatch;
    });
  }, [searchTerm, postalCode, region, province, municipality, barangay, distance, tuition, commuteBuckets, subsidies, schoolTypes, religious]);

  const metrics = useMemo(() => {
    const escCount = filteredSchools.filter((school) => school.type === "private_esc").length;
    const avgNet = filteredSchools.length === 0 ? 0 : Math.round(filteredSchools.reduce((sum, school) => sum + school.net_cost, 0) / filteredSchools.length);
    return { escCount, avgNet };
  }, [filteredSchools]);

  const toggleArray = (setter, values, value) => {
    setter(values.includes(value) ? values.filter((item) => item !== value) : [...values, value]);
  };

  const clearAll = () => {
    setSearchTerm(""); setPostalCode(""); setRegion(""); setProvince(""); setMunicipality(""); setBarangay(""); setDistance([0, 100]); setTuition([0, 250000]); setCommuteBuckets([]); setSubsidies([]); setSchoolTypes([]); setReligious([]); setSelectedSchool(schools[0]);
  };

  const selectSchool = (school) => {
    setSelectedSchool(school);
    if (school) setActiveView("Student View");
  };

  const tabs = ["Student View", "School View", "DepEd View"];
  const comingSoon = activeView === "Student View" ? null : activeView;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white font-['SF_Pro_Text',-apple-system,BlinkMacSystemFont,'Segoe_UI',system-ui,sans-serif] text-[#1a1d23]">
      <aside className="flex h-full w-[340px] xl:w-[440px] shrink-0 flex-col border-r border-[#e2e4e9] bg-[#f8f9fa] z-20 shadow-[4px_0_24px_rgba(0,0,0,0.02)]">
        <header className="border-b border-[#e2e4e9] bg-white px-4 py-4 sm:px-6 sm:py-5 z-10">
          <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-12 w-[76px] items-center justify-center rounded-xl border border-[#e2e4e9] bg-white px-2 shadow-[0_1px_3px_rgba(0,0,0,0.06)] sm:w-32">
                <img src="/assets/ecair-logo.png" alt="ECAIR" className="max-h-9 w-full object-contain" />
              </div>
              <div className="flex h-12 w-[76px] items-center justify-center rounded-xl border border-[#e2e4e9] bg-white px-2 shadow-[0_1px_3px_rgba(0,0,0,0.06)] sm:w-32">
                <img src="/assets/deped-logo.png" alt="DepEd" className="max-h-10 w-full object-contain" />
              </div>
            </div>
            <div className="hidden rounded-full border border-[#e2e4e9] bg-[#f8f9fb] px-3 py-1.5 text-xs font-semibold text-[#6b7280] sm:block">
              Executive Demo
            </div>
          </div>
          <div className="mt-5">
            <h1 className="font-['SF_Pro_Display',-apple-system,BlinkMacSystemFont,'Segoe_UI',system-ui,sans-serif] text-2xl font-bold tracking-normal text-[#1a1d23] sm:text-3xl">
              PAARAL
            </h1>
            <p className="mt-1 text-[13px] leading-5 text-[#6b7280]">
              Platform for Analyzing Access and Resource Allocation in Learning
            </p>
          </div>
        </header>

        <div className="border-b border-[#e2e4e9] bg-white px-4 py-4 sm:px-6 z-10">
          <div className="grid grid-cols-3 gap-2">
            {tabs.map((tab) => (
              <button
                key={tab}
                type="button"
                title={tab === "Student View" ? "Current view" : "Coming Soon"}
                onClick={() => setActiveView(tab)}
                className={`h-10 rounded-full text-[13px] font-semibold transition focus:outline-none ${
                  activeView === tab
                    ? "bg-[#1a4b8c] text-white shadow-[0_4px_12px_rgba(26,75,140,0.22)]"
                    : "border border-[#e2e4e9] bg-white text-[#1a4b8c] hover:bg-[#f8f9fb]"
                }`}
              >
                {tab.replace(" View", "")}
              </button>
            ))}
          </div>

          <div className="mt-4 flex gap-2">
            <div className="relative flex-1 min-w-0">
              <Search className="absolute left-3 top-3.5 h-4 w-4 text-[#9ca3af]" />
              <input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search for schools or keywords..."
                className="h-11 w-full rounded-xl border border-[#e2e4e9] bg-white pl-10 pr-3 text-[15px] outline-none transition placeholder:text-[#9ca3af] focus:border-[#1a4b8c] focus:ring-2 focus:ring-[#1a4b8c]/10 truncate"
              />
            </div>
            <button
              type="button"
              onClick={() => setActiveView("Student View")}
              className="flex h-11 shrink-0 items-center gap-2 rounded-xl bg-[#1a4b8c] px-3 text-sm font-bold text-white transition hover:bg-[#143b70] focus:outline-none sm:px-4"
            >
              <Search className="h-4 w-4" />
              <span className="hidden sm:inline">Search</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 sm:gap-3 px-4 py-5 sm:px-6 z-10">
          <div className="rounded-2xl border border-[#e2e4e9] bg-white p-2.5 sm:p-3 shadow-sm min-w-0 flex flex-col justify-center">
            <p className="text-[10px] sm:text-[11px] uppercase tracking-[0.05em] text-[#9ca3af] leading-tight whitespace-nowrap">Options</p>
            <p className="mt-1 text-lg sm:text-xl font-bold tracking-tight text-[#1a1d23]">{filteredSchools.length}</p>
          </div>
          <div className="rounded-2xl border border-[#e2e4e9] bg-white p-2.5 sm:p-3 shadow-sm min-w-0 flex flex-col justify-center">
            <p className="text-[10px] sm:text-[11px] uppercase tracking-[0.05em] text-[#9ca3af] leading-tight whitespace-nowrap">ESC Slots</p>
            <p className="mt-1 text-lg sm:text-xl font-bold tracking-tight text-[#449e52]">{metrics.escCount}</p>
          </div>
          <div className="rounded-2xl border border-[#e2e4e9] bg-white p-2.5 sm:p-3 shadow-sm min-w-0 flex flex-col justify-center">
            <p className="text-[10px] sm:text-[11px] uppercase tracking-[0.05em] text-[#9ca3af] leading-tight whitespace-nowrap">Avg Cost</p>
            <p className="mt-1 text-[15px] sm:text-lg font-bold tracking-tight text-[#1a4b8c] whitespace-nowrap">{pesos(metrics.avgNet)}</p>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-5 sm:px-6 custom-scrollbar">
          <section className="rounded-2xl border border-[#e2e4e9] bg-white shadow-sm overflow-hidden">
            <div className="flex items-center justify-between border-b border-[#e2e4e9] px-5 py-4 bg-white sticky top-0 z-10">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-[18px] w-[18px] text-[#1a4b8c]" />
                <h2 className="text-[17px] font-bold text-[#1a1d23]">Criteria</h2>
              </div>
              <button type="button" onClick={clearAll} className="text-sm font-bold text-[#1a4b8c] hover:underline focus:outline-none">
                Clear all
              </button>
            </div>

            <div className="px-5">
              <FilterSection
                title="Location"
                id="location"
                open={openSections.location}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                <input
                  value={postalCode}
                  onChange={(event) => setPostalCode(event.target.value)}
                  placeholder="Postal code or barangay keyword"
                  className="h-11 w-full rounded-xl border border-[#e2e4e9] px-4 text-[15px] outline-none transition placeholder:text-[#9ca3af] focus:border-[#1a4b8c] focus:ring-2 focus:ring-[#1a4b8c]/10 mb-3"
                />
                <div className="grid grid-cols-2 gap-3">
                  <SelectField value={region} onChange={setRegion} options={regionOptions} placeholder="Region" />
                  <SelectField value={province} onChange={setProvince} options={provinceOptions} placeholder="Province" />
                  <SelectField value={municipality} onChange={setMunicipality} options={municipalityOptions} placeholder="Municipality" />
                  <SelectField value={barangay} onChange={setBarangay} options={barangayOptions} placeholder="Barangay" />
                </div>
              </FilterSection>

              <FilterSection
                title="Distance"
                id="distance"
                open={openSections.distance}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                <RangePair min={0} max={100} value={distance} onChange={setDistance} format={(value) => `${value} km`} />
              </FilterSection>

              <FilterSection
                title="Commute"
                id="commute"
                open={openSections.commute}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                <CheckboxRow
                  checked={commuteBuckets.includes("under5")}
                  label="<5 minutes"
                  sublabel="nearby"
                  onChange={() => toggleArray(setCommuteBuckets, commuteBuckets, "under5")}
                />
                <CheckboxRow
                  checked={commuteBuckets.includes("15to30")}
                  label="15-30 minutes"
                  sublabel="short ride"
                  onChange={() => toggleArray(setCommuteBuckets, commuteBuckets, "15to30")}
                />
                <CheckboxRow
                  checked={commuteBuckets.includes("over30")}
                  label="30+ minutes"
                  sublabel="regional"
                  onChange={() => toggleArray(setCommuteBuckets, commuteBuckets, "over30")}
                />
              </FilterSection>

              <FilterSection
                title="Tuition Fees"
                id="tuition"
                open={openSections.tuition}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                <RangePair
                  min={0}
                  max={250000}
                  step={5000}
                  value={tuition}
                  onChange={setTuition}
                  format={(value) => pesos(value).replace(".00", "")}
                />
              </FilterSection>

              <FilterSection
                title="ESC Subsidy Amount"
                id="subsidy"
                open={openSections.subsidy}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                {[9000, 11000, 13000].map((amount) => (
                  <CheckboxRow
                    key={amount}
                    checked={subsidies.includes(amount)}
                    label={pesos(amount)}
                    sublabel="annual"
                    onChange={() => toggleArray(setSubsidies, subsidies, amount)}
                  />
                ))}
              </FilterSection>

              <FilterSection
                title="School Type"
                id="type"
                open={openSections.type}
                onToggle={(id) => setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                <div className="space-y-1">
                  <p className="mb-2 px-2 text-[11px] font-bold uppercase tracking-wider text-[#9ca3af]">Sector</p>
                  <CheckboxRow
                    checked={schoolTypes.includes("public")}
                    label="Public"
                    onChange={() => toggleArray(setSchoolTypes, schoolTypes, "public")}
                  />
                  <CheckboxRow
                    checked={schoolTypes.includes("private_no_esc")}
                    label="Private no ESC"
                    onChange={() => toggleArray(setSchoolTypes, schoolTypes, "private_no_esc")}
                  />
                  <CheckboxRow
                    checked={schoolTypes.includes("private_esc")}
                    label="Private with ESC"
                    onChange={() => toggleArray(setSchoolTypes, schoolTypes, "private_esc")}
                  />
                </div>
                <div className="mt-5 space-y-1 pb-1">
                  <p className="mb-2 px-2 text-[11px] font-bold uppercase tracking-wider text-[#9ca3af]">Religious Affiliation</p>
                  <CheckboxRow
                    checked={religious.includes("sectarian")}
                    label="Sectarian"
                    onChange={() => toggleArray(setReligious, religious, "sectarian")}
                  />
                  <CheckboxRow
                    checked={religious.includes("non_sectarian")}
                    label="Non-Sectarian"
                    onChange={() => toggleArray(setReligious, religious, "non_sectarian")}
                  />
                </div>
              </FilterSection>
            </div>
          </section>

          <section className="mt-6">
            <div className="mb-4 flex items-center justify-between px-1">
              <h2 className="text-[17px] font-bold text-[#1a1d23]">Results List</h2>
              <span className="rounded-full bg-[#e2e4e9] px-3 py-1 text-xs font-bold text-[#6b7280]">
                {filteredSchools.length} schools
              </span>
            </div>
            <div className="space-y-4 pb-8">
              {filteredSchools.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-[#d1d5db] bg-white p-6 text-center">
                  <Info className="mx-auto h-6 w-6 text-[#9ca3af]" />
                  <p className="mt-2 text-sm font-semibold text-[#1a1d23]">No matching schools</p>
                  <p className="mt-1 text-xs text-[#6b7280]">Clear filters to widen the options shown to this household.</p>
                </div>
              ) : (
                filteredSchools.map((school) => (
                  <ResultCard
                    key={school.id}
                    school={school}
                    selected={selectedSchool?.id === school.id}
                    onSelect={selectSchool}
                  />
                ))
              )}
            </div>
          </section>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col h-full relative">
        <div className="flex-1 relative min-h-0 bg-[#e9f0f5]">
          <PhilippinesMap
            filteredSchools={filteredSchools}
            selectedSchool={selectedSchool}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onSelect={selectSchool}
            comingSoon={comingSoon}
          />
        </div>
      </main>
    </div>
  );
}
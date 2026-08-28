import type { School } from "@/types/school";
import { netFeesLabel, pesos, titleCase } from "@/lib/schools";
import FactTile from "@/components/molecules/FactTile";

interface SchoolFactSheetProps {
  school: School;
}

/** Detail-page fact grid — location, fees, and ESC subsidy slots. Fields
 * that are null in the real BigQuery-sourced data show "Not available"
 * rather than being hidden, so gaps in the dataset are visible, not
 * papered over. */
export default function SchoolFactSheet({ school }: SchoolFactSheetProps) {
  return (
    <div className="flex flex-col gap-6">
      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-widest text-slate-500">
          Location
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <FactTile
            label="Municipality"
            value={titleCase(school.municipality) ?? "Not available"}
          />
          <FactTile
            label="Barangay"
            value={titleCase(school.barangay) ?? "Not available"}
          />
          <FactTile
            label="Setting"
            value={
              school.urban_rural === "U"
                ? "Urban"
                : school.urban_rural === "R"
                  ? "Rural"
                  : "Not available"
            }
          />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-widest text-slate-500">
          Fees
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <FactTile label="Tuition" value={pesos(school.esc_tuition)} />
          <FactTile label="Other Fees" value={pesos(school.esc_other_fees)} />
          <FactTile
            label="Miscellaneous Fees"
            value={pesos(school.esc_misc_fees)}
          />
          <FactTile
            label="Total Fees"
            value={pesos(school.esc_total_fees)}
          />
          <FactTile
            label="Subsidy Amount"
            value={pesos(school.esc_subsidy_amount)}
          />
          <FactTile
            label="Net Fees (after subsidy)"
            value={netFeesLabel(school.esc_net_fees)}
          />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-widest text-slate-500">
          Subsidy Slots
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <FactTile
            label="Total Slots"
            value={school.slot_total ?? "Not available"}
          />
          <FactTile
            label="Slots Unutilized"
            value={school.slot_unutilized ?? "Not available"}
          />
        </div>
      </section>
    </div>
  );
}

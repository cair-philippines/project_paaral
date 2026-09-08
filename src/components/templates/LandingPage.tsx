import HeroSection from "@/components/organisms/HeroSection";
import AboutEscSection from "@/components/organisms/AboutEscSection";
import AboutPaaralSection from "@/components/organisms/AboutPaaralSection";
import ReassuranceSection from "@/components/organisms/ReassuranceSection";
import BeforeYouStartSection from "@/components/organisms/BeforeYouStartSection";
import GuidedProcessSection from "@/components/organisms/GuidedProcessSection";
import ScrollToHash from "@/components/atoms/ScrollToHash";

/** The landing page deliberately has no SiteHeader/navbar — HeroSection
 * already carries its own "Browse Schools" CTA, so the only nav element
 * that needs to exist here is a floating Log In / My Account control.
 * AboutEsc/AboutPaaral are the scroll targets for the hero's "Know More"
 * buttons — placed directly after the hero, in the same order as the
 * buttons, on plain white/slate-50 backgrounds (the gradient stays
 * scoped to the hero only, per Paula's explicit call).
 *
 * `ScrollToHash` (2026-09-08) handles landing here with a `#about-esc`/
 * `#about-paaral` hash from another page's nav link (e.g. SiteHeader on
 * `/browse`) — the browser's own hash-anchor scroll doesn't reliably fire
 * on that cross-route case in this app. */
export default function LandingPage() {
  return (
    <div className="flex flex-1 flex-col">
      <ScrollToHash />
      <HeroSection />
      <AboutEscSection />
      <AboutPaaralSection />
      <ReassuranceSection />
      <BeforeYouStartSection />
      <GuidedProcessSection />
    </div>
  );
}

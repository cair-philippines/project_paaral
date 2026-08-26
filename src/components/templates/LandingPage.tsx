import HeroSection from "@/components/organisms/HeroSection";
import ReassuranceSection from "@/components/organisms/ReassuranceSection";
import BeforeYouStartSection from "@/components/organisms/BeforeYouStartSection";
import GuidedProcessSection from "@/components/organisms/GuidedProcessSection";

/** The landing page deliberately has no SiteHeader/navbar — HeroSection
 * already carries its own "Browse Schools" CTA, so the only nav element
 * that needs to exist here is a floating Log In / My Account control. */
export default function LandingPage() {
  return (
    <div className="flex flex-1 flex-col">
      <HeroSection />
      <ReassuranceSection />
      <BeforeYouStartSection />
      <GuidedProcessSection />
    </div>
  );
}

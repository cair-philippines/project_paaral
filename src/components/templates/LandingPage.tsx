import SiteHeader from "@/components/organisms/SiteHeader";
import HeroSection from "@/components/organisms/HeroSection";
import ReassuranceSection from "@/components/organisms/ReassuranceSection";
import BeforeYouStartSection from "@/components/organisms/BeforeYouStartSection";
import GuidedProcessSection from "@/components/organisms/GuidedProcessSection";

export default function LandingPage() {
  return (
    <div className="flex flex-1 flex-col">
      <SiteHeader />
      <HeroSection />
      <ReassuranceSection />
      <BeforeYouStartSection />
      <GuidedProcessSection />
    </div>
  );
}

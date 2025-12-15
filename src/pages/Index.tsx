import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import CareerTimeline from "@/components/CareerTimeline";
import BlogSection from "@/components/BlogSection";
import ResearchProjects from "@/components/ResearchProjects";
import EducationSection from "@/components/EducationTimeline";
import Personal from "@/components/Personal";
import Volunteering from "@/components/Volunteering";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <HeroSection />
      <ResearchProjects />
      <CareerTimeline />
      <EducationSection />
      <Volunteering />
      <Personal />
      {/* <BlogSection /> */}
    </div>
  );
};

export default Index;
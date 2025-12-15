import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Github, Instagram, Linkedin, Mail, Twitter } from "lucide-react";
import meImage from "@/assets/image.png";

const HeroSection = () => {
  return (
    <section id="hero" className="min-h-screen flex items-center justify-center bg-hero-gradient">
      <div className="container mx-auto px-6 py-20">
        <div className="max-w-4xl mx-auto text-center">
          {/* Profile Image */}
          <div className="mb-8 animate-fade-in">
            <div className="relative inline-block">
              <img
                src={meImage}
                alt="AI Developer Portrait"
                className="w-32 h-32 md:w-40 md:h-40 rounded-full object-cover border-4 border-primary/20 shadow-glow mx-auto"
              />
              <div className="absolute inset-0 rounded-full bg-ai-gradient opacity-10 animate-glow"></div>
            </div>
          </div>

          {/* Name and Title */}
          <div className="mb-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <h1 className="text-4xl md:text-6xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-2">
              Alexandra Schild
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground">
              AI Researcher & Data Scientist
            </p>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap justify-center gap-2 mb-8 animate-fade-in" style={{ animationDelay: '0.4s' }}>
            <Badge variant="secondary">Multimodal LLMs</Badge>
            <Badge variant="secondary">Computer Vision</Badge>
            <Badge variant="secondary">Representation Learning</Badge>
            <Badge variant="secondary">Statistics</Badge>
            <Badge variant="secondary">Data Science</Badge>
          </div>

          {/* Bio */}
          <div className="mb-8 animate-fade-in" style={{ animationDelay: '0.6s' }}>
            <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              I'm a second year PhD Student in AIIS Lab at Hasso-Plattner-Institute.
              My current research focuses on improving VLM's ability to understand highly detailed scenes, spacial relationships and geometries.
              Previously, I worked as a Data Scientist on a variety of exciting problems spanning from banking and fintech to fashion, design and architecture.
            </p>
          </div>

          {/* Social Links */}
          <div className="flex justify-center gap-4 animate-fade-in" style={{ animationDelay: '0.8s' }}>
            <a
              href="https://github.com/alexandraschild"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" size="icon" className="hover:shadow-glow transition-all">
                <Github className="h-5 w-5" />
                <span className="sr-only">GitHub</span>
              </Button>

            </a>
            <a
              href="https://www.linkedin.com/in/alexandra-schild/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" size="icon" className="hover:shadow-glow transition-all">
                <Linkedin className="h-5 w-5" />
                <span className="sr-only">LinkedIn</span>
              </Button>
            </a>
            <a
              href="https://www.instagram.com/alexandra_kuda/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" size="icon" className="hover:shadow-glow transition-all">
                <Instagram className="h-5 w-5" />
                <span className="sr-only">Instagram</span>
              </Button>
            </a>
            <Button
              variant="outline"
              size="icon"
              className="hover:shadow-glow transition-all"
              asChild
            >
              <a href="mailto:me@alexandrakudaeva.com">
                <Mail className="h-5 w-5" />
                <span className="sr-only">Email</span>
              </a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
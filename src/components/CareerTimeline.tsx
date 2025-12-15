import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Briefcase, GraduationCap } from "lucide-react";

interface TimelineItem {
  id: string;
  type: "work" | "education";
  title: string;
  company: string;
  location: string;
  period: string;
  description: string[];
  technologies: string[];
}

const timelineData: TimelineItem[] = [
  {
    id: "1",
    type: "work",
    title: "Senior Data Scientist",
    company: "H&M Group",
    location: "Berlin, Germany",
    period: "Aug 2022 - Aug 2024",
    description: ["Led development of internal visual search engine",
      "Improved existing similarity and compatibility recommendations models",
      "Created pipeline for automatic metadata generation for digital prints",
      "Acted as a technical expert for new AI initiatives in H&M",
      "Created an interactive model recommendations visualization tool in Streamlit"
    ],
    technologies: ["PyTorch", "Spark", "GCP", "Databricks", "BigQuery", "CICD"]
  },
  {
    id: "2",
    type: "work",
    title: "Tech Founder",
    company: "Palasts / MeinHaus.Digital",
    location: "Berlin, Germany",
    period: "July 2021 - Dec 2024",
    description: [
      "Developed AI and software development strategy",
      "Led a software development team (2 developers)",
      "Developed and deployed ML (Computer Vision, LLM) pipelines to support design, architecture, and energy consulting processes (DigitalOcean, GCP)",
      "Designed data collection systems and created synthetic training datasets"
    ],
    technologies: ["PyTorch", "Digital Ocean", "PostGres", "Next.js"]
  },
  {
    id: "3",
    type: "work",
    title: "Data Scientist",
    company: "Klarna",
    location: "Berlin, Germany",
    period: "Apr 2021 - Jul 2022",
    description: ["Developed and deployed to production credit risk/abuse models based on large amounts of transactional and customer data (XGBoost, CatBoost) used for the US/UK markets (affected more than 20M users)",
      "Created an interactive explainability tool to explain model decisions to stakeholders and tackle discrimination and model biases",
      "Performed various types of analysis resulting in model improvement and ensuring ethical usage of AI"
    ],
    technologies: ["Python", "Scikit-learn", "AWS", "MLflow"]
  },
  {
    id: "4",
    type: "work",
    title: "Senior Consultant",
    company: "KPMG",
    location: "Berlin, Germany / Moscow, Russia",
    period: "May 2016 - Apr 2021",
    description: [
      "Developed credit risk models (PD) for large retail and corporate portfolios in compliance with regulatory requirements ensuring stability of banking sector (loss modeling)",
      "Validated predictive statistical models (PD, LGD) for large retail portfolio",
      "Automated management reporting in R Markdown (yearly validation, monitoring)",
      "Developed automatic data quality monitoring framework that allowed prompt identification of DQ related problems and system failures",
      "Organized external workshops (topics: Statistical Modeling, Credit Risk Modeling)"
    ],
    technologies: ["Python", "R", "SAS", "SQL"]
  },
  /*  {
      id: "5",
      type: "work",
      title: "Risk Analytics Intern",
      company: "VTB Bank",
      location: "Moscow, Russia",
      period: "Summer 2015",
      description: ["Dummy test description"],
      technologies: ["AI/ML", "Research", "Neural Networks", "Optimization"]
    },*/
  {
    id: "6",
    type: "work",
    title: "Market Risk and Derivatives Intern",
    company: "Moscow Exchange",
    location: "Moscow, Russia",
    period: "Summer 2014",
    description: ["Creating Datasets from Bloomberg Terminal", "Analyzing Time-Series Data"],
    technologies: ["AI/ML", "Research", "Neural Networks", "Optimization"]
  }
];

const CareerTimeline = () => {
  return (
    <section id="career" className="py-10 bg-background">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto">
          {/* Section Header */}
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl md:text-4xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-4">
              Industry Experience
            </h2>
            <p className="text-lg text-muted-foreground">
              My path through the world of AI and machine learning
            </p>
          </div>

          {/* Timeline */}
          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary via-accent to-primary/30"></div>

            {/* Timeline Items */}
            <div className="space-y-8">
              {timelineData.map((item, index) => (
                <div
                  key={item.id}
                  className="relative animate-slide-up"
                  style={{ animationDelay: `${index * 0.2}s` }}
                >
                  {/* Timeline Icon */}
                  <div className="absolute left-6 w-5 h-5 rounded-full bg-primary border-4 border-background shadow-glow flex items-center justify-center">
                    {item.type === "work" ? (
                      <Briefcase className="w-3 h-3 text-primary-foreground" />
                    ) : (
                      <GraduationCap className="w-3 h-3 text-primary-foreground" />
                    )}
                  </div>

                  {/* Content Card */}
                  <div className="ml-16">
                    <Card className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow">
                      <CardContent className="p-6">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-3">
                          <div>
                            <h3 className="text-xl font-semibold text-foreground mb-1">
                              {item.title}
                            </h3>
                            <p className="text-primary font-medium">
                              {item.company} • {item.location}
                            </p>
                          </div>
                          <Badge variant="outline" className="mt-2 md:mt-0 w-fit">
                            {item.period}
                          </Badge>
                        </div>
                        {item.description.map((item) => (
                          <p className="text-muted-foreground mb-0 leading-relaxed">
                            • {item}
                          </p>
                        ))}

                        <div className="flex flex-wrap gap-2 mt-4">
                          {item.technologies.map((tech) => (
                            <Badge key={tech} variant="secondary" className="text-xs">
                              {tech}
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CareerTimeline;
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GraduationCap, Award, BookOpen } from "lucide-react";

const educationData = [
    {
        degree: "Ph.D. in Computer Science",
        institution: "Hasso Plattner Institute, University of Potsdam",
        location: "Potsdam, Germany",
        period: "2024 - 2027 (Expected)",
        gpa: "",
        thesis: "Efficient Neural Architecture Search for Resource-Constrained Environments",
        advisor: "Prof. Gerard de Melo",
        coursework: [""],
        achievements: ["HPI Research Grant", "HPI-MIT Grant: Design for Sustainability"]
    },
    {
        degree: "M.S. in Statistics",
        institution: "Techical University of Berlin (TUB), Humboldt University of Berlin (HU)",
        location: "Berlin, Germany",
        period: "2018 - 2021",
        gpa: "1.7/1.0",
        thesis: "Multilingual Relationship Extraction with BERT",
        advisor: "Prof. Alan Akbik",
        coursework: ["Machine Learning", "Econometrics", "Stochastic Analysis", "Statistical Inference"],
        achievements: []
    },
    {
        degree: "Exchange Semester",
        institution: "Technical University of Munich (TUM)",
        location: "Munich, Germany",
        period: "2015 - 2016",
        thesis: "",
        gpa: "",
        advisor: "",
        coursework: ["Multidimentional Modelling with Copula Functions", "Quantitative Finance", "Stochastic Analysis"],
        achievements: ["DAAD Scholarship"]
    },
    {
        degree: "B.S. in Economics & Probability Theory",
        institution: "Lomonosov Moscow State University (MSU)",
        location: "Moscow, Russia",
        period: "2012 - 2016",
        gpa: "4.5/5.0",
        advisor: "Prof. Evgenii Lukash",
        thesis: "Currency Risk Modelling with Copula-Functions",
        coursework: ["Probability Theory", "Linear Algebra", "Statistics", "Mathematical Analysis", "Micro- & Macroeconomics"],
        achievements: ["Academic Excellence Scholarship", "Grant for a Sustainable Idea Implementation", "Nomination for Best Thesis Award"]
    }
];

const EducationSection = () => {
    return (
        <section id="education" className="py-20 px-4">
            <div className="container mx-auto max-w-4xl">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                        Education
                    </h2>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        Academic foundation and continuous learning in artificial intelligence and computer science
                    </p>
                </div>

                {/* Education */}
                <div className="space-y-6 mb-12">
                    {educationData.map((edu, index) => (
                        <Card key={index} className="hover:shadow-lg transition-all duration-300">
                            <CardHeader>
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-primary/10 rounded-lg">
                                        <GraduationCap className="w-6 h-6 text-primary" />
                                    </div>
                                    <div className="flex-1">
                                        <CardTitle className="text-xl mb-1">{edu.degree}</CardTitle>
                                        <CardDescription className="text-base">
                                            <span className="font-semibold">{edu.institution}</span> • {edu.location}
                                        </CardDescription>
                                        <div className="flex gap-4 mt-2 text-sm text-muted-foreground">
                                            <span>{edu.period}</span>
                                            {edu.gpa !== "" && <span>GPA: {edu.gpa}</span>}
                                        </div>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {edu.thesis !== "" && <div>
                                    <h4 className="font-semibold mb-2">Thesis</h4>
                                    <p className="text-sm text-muted-foreground italic">"{edu.thesis}"</p>
                                    <p className="text-sm text-muted-foreground mt-1">Advisor: {edu.advisor}</p>
                                </div>}


                                <div>
                                    <h4 className="font-semibold mb-2">Key Coursework</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {edu.coursework.map((course) => (
                                            <Badge key={course} variant="outline" className="text-xs">
                                                {course}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                                {edu.achievements.length !== 0 &&
                                    <div>
                                        <h4 className="font-semibold mb-2 flex items-center gap-2">
                                            <Award className="w-4 h-4" />
                                            Achievements
                                        </h4>
                                        <ul className="text-sm text-muted-foreground space-y-1">
                                            {edu.achievements.map((achievement, idx) => (
                                                <li key={idx} className="flex items-center gap-2">
                                                    <span className="w-1 h-1 bg-primary rounded-full"></span>
                                                    {achievement}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>}
                            </CardContent>
                        </Card>
                    ))}
                </div>

                {/* Certifications */}
                {/* <div>
                    <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                        <BookOpen className="w-6 h-6 text-primary" />
                        Professional Certifications
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                        {certifications.map((cert, index) => (
                            <Card key={index} className="p-4">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h4 className="font-semibold">{cert.name}</h4>
                                        <p className="text-sm text-muted-foreground">{cert.issuer}</p>
                                    </div>
                                    <Badge variant="secondary">{cert.year}</Badge>
                                </div>
                            </Card>
                        ))}
                    </div>
                </div>*/}
            </div>
        </section>
    );
};

export default EducationSection;
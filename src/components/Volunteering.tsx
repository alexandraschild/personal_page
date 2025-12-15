import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GraduationCap, Award, BookOpen } from "lucide-react";


const Volunteering = () => {
    const volunteering = [
        {
            organization: "TUMO",
            role: "Workshop Instructor",
            description: "Teaching Python, Computer Vision and GenAI to teens in Armenia, France and Germany",
            duration: "2022 - Present"
        },
        {
            organization: "Omdena",
            role: "Project Lead for Berlin Chapter",
            description: "Organized AI projects for education and social good",
            duration: "2022 - 2023"
        },
        {
            organization: "Software Development Academy",
            role: "Course Lead & Mentor",
            description: "Python and Data Science courses for career changers: Statistical Machine Learning, Data Visualization, Computer Vision",
            duration: "2021 - 2023"
        }
    ];
    return (
        <section id="personal" className="py-10 px-4 bg-muted/20">
            <div className="container mx-auto max-w-6xl">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                        Teaching & Community Involvement
                    </h2>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        I love the feeling that teaching gives me, it's the deepest understanding of the subject you can never achieve on your own. I particularly enjoy working with teens and trying to give my best mentoring Bachelor and Master thesis projects.
                    </p>
                </div>
                <div className="grid gap-4">
                    {volunteering.map((vol, index) => (
                        <div key={index} className="flex justify-between items-start p-3 border rounded-lg">
                            <div>
                                <h4 className="font-semibold">{vol.organization}</h4>
                                <p className="text-sm text-muted-foreground">{vol.role}</p>
                                <p className="text-xs text-muted-foreground mt-1">{vol.description}</p>
                            </div>
                            <Badge variant="outline" className="text-xs">
                                {vol.duration}
                            </Badge>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Volunteering;
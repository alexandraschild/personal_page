import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, Github } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface ResearchProject {
    id: string;
    title: string;
    excerpt: string;
    status: string;
    collaborators: string[];
    publications: string[];
    tags: string[];
    url: string;
    links: {
        paper?: string;
        code?: string;
        [key: string]: string | undefined; // allows flexibility for future links
    };
}

const researchProjects: ResearchProject[] = [

    {
        id: "1",
        title: "AI-Empowered Architect: VLMs for Design of Energy-Efficient Structures",
        excerpt: "We explore the use of VLMs to assist architects in designing physically valid structures that align with desired emotional or aesthetic qualities. By learning disentangled representations of geometry, performance, and affective impact, we aim to bridge spatial reasoning, physical grounding, and emotional design in computational architecture.",
        status: "active",
        collaborators: ["MIT Architecture"],
        publications: [""],
        tags: ["VLMs", "Affectionate Computing", "Geometry Understanding and Generation", "Spatial Reasoning"],
        url: "/research/1",
        links: {
            paper: "",
            code: ""
        }
    },
    {
        id: "2",
        title: "Evaluating VLMs for High-Resolution and Detail-Rich Visual Understanding",
        excerpt: "We investigate the limitations of current vision–language models (VLMs) when processing high-resolution images or small, detail-rich regions. By developing a systematic evaluation framework and novel metrics, we assess model scalability and reveal how architectural bottlenecks impact performance on fine-grained visual reasoning tasks.",
        status: "active",
        collaborators: [],
        publications: [""],
        tags: ["VLMs", "High-Resolution Vision", "Fine-Grained Visual Reasoning", "Benchmarking"],
        url: "/research/2",
        links: {
            paper: "https://openreview.net/pdf?id=u6DbnUgBDV",
            code: ""
        }
    },
    {
        id: "3",
        title: "Grounding Semantically Complex Regions in Visual Scenes",
        excerpt: "We study how vision–language models can holistically ground semantically complex and relational concepts in images. Our research aims to move beyond object-level understanding toward unified spatial, geometric, and relational reasoning over detailed real-world scenes.",
        status: "active",
        collaborators: ["MIT City Labs"],
        publications: ["AAAI26 AI for Social Impact (accepted)"],
        tags: ["VLMs", "Spatial Reasoning", "Region Detection", "Semantic Grounding"],
        url: "/research/3",
        links: {
            paper: "https://arxiv.org/abs/2509.13484",
            code: "https://github.com/lyons66/MINGLE"
        },

    },
];

const ResearchProjects = () => {
    const navigate = useNavigate();
    return (
        <section id="researchprojects" className="py-10 bg-secondary/30">
            <div className="container mx-auto px-6">
                <div className="max-w-6xl mx-auto">
                    {/* Section Header */}
                    <div className="text-center mb-16 animate-fade-in">
                        <h2 className="text-3xl md:text-4xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-4">
                            Current Research Projects
                        </h2>
                        <p className="text-lg text-muted-foreground mb-8">
                            My research focuses on the advancement and understanding of multimodal models.
                        </p>
                    </div>

                    {/* Research Projects Grid */}
                    <div className="grid md:grid-cols-3 gap-6">
                        {researchProjects.map((post, index) => (
                            <Card
                                key={post.id}
                                className="flex flex-col h-full bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow animate-slide-up"
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <CardHeader>
                                    <div className="flex justify-between items-start mb-2">
                                        <CardTitle className="text-xl">{post.title}</CardTitle>
                                        <Badge
                                            variant={post.status === "Active" ? "default" : "secondary"}
                                            className={post.status === "Active" ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300" : ""}
                                        >
                                            {post.status}
                                        </Badge>
                                    </div>
                                </CardHeader>
                                <CardContent className="flex flex-col flex-grow">
                                    <p className="text-muted-foreground mb-4 leading-relaxed">
                                        {post.excerpt}
                                    </p>
                                    <div className="flex flex-wrap gap-2 mb-4 mt-auto">
                                        {post.tags.map((tag) => (
                                            <Badge key={tag} variant="secondary" className="text-xs">
                                                {tag}
                                            </Badge>
                                        ))}
                                    </div>
                                    <div className="flex gap-2 pt-2">
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            className="p-0 h-auto text-primary hover:text-primary/80"
                                            onClick={() => navigate(`/research/${post.id}`)}
                                        >
                                            Read More <ExternalLink className="ml-1 h-3 w-3" />
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>

                    <div className="text-center my-8 animate-fade-in">
                        <Button
                            variant="outline"
                            size="lg"
                            onClick={() => navigate('/research')}
                            className="hover:shadow-glow transition-all"
                        >
                            View All Research Projects <ExternalLink className="ml-2 h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default ResearchProjects;
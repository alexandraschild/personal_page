import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ExternalLink, Search, ArrowLeft } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

interface BlogPost {
    id: string;
    title: string;
    excerpt: string;
    status: string;
    collaborators: string[];
    publications: string[];
    content: string;
    tags: string[];
    url: string;
    links: {
        paper?: string;
        code?: string;
        [key: string]: string | undefined; // allows flexibility for future links
    };
}

const blogPosts: BlogPost[] = [
    {
        id: "1",
        title: "AI-Empowered Architect",
        excerpt: "We explore the use of vision–language models to assist architects in designing physically valid structures that align with desired emotional or aesthetic qualities. By learning disentangled representations of geometry, performance, and affective impact, our approach aims to bridge spatial reasoning, physical grounding, and emotional design in computational architecture.",
        status: "active",
        collaborators: ["MIT Architecture"],
        publications: [""],
        tags: ["VLMs", "Affectionate Computing", "Geometry Understanding and Generation", "Spatial Reasoning"],
        url: "#",
        content: "",
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
        url: "#",
        content: "",
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
        url: "#",
        content: "",
        links: {
            paper: "https://arxiv.org/abs/2509.13484",
            code: "https://github.com/lyons66/MINGLE"
        },

    },
    {
        id: "4",
        title: "Faithful Feature Localization in ViT using Attention-LRP",
        excerpt: "I supervised a bachelor research project investigating how Vision Transformers localize discriminative features in fine-grained re-identification. By adapting layer-wise relevance propagation to metric learning with ViTs, this work provides faithful, quantitative explanations of model decisions and formed the basis for a joint publication.",
        status: "complete",
        collaborators: ["Iven Schlegelmilch, Max Schall"],
        publications: ["WACV26 (accepted)"],
        tags: ["Vision Transformers", "Interpretability", "Layer-wise Relevance Propagation", "Fine-Grained Recognition"],
        url: "#",
        content: "",
        links: {
            paper: "https://www.arxiv.org/abs/2512.07776",
            code: "https://gorilla-watch.github.io/"
        }
    },
];

const Research = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-background pt-20">
            <div className="container mx-auto px-6 py-12">
                {/* Header */}
                <div className="mb-12 animate-fade-in">
                    <Button
                        variant="ghost"
                        onClick={() => navigate('/')}
                        className="mb-6 hover:bg-secondary/50"
                    >
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Home
                    </Button>

                    <div className="text-center mb-8">
                        <h1 className="text-4xl md:text-5xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-4">
                            Research Projects
                        </h1>
                        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                            My research focuses on the advancement and understanding of multimodal models.
                        </p>
                    </div>
                </div>


                {/* All Posts */}
                <div>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {blogPosts.map((post, index) => (
                            <Card
                                key={post.id}
                                className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow animate-slide-up"
                                style={{ animationDelay: `${(blogPosts.length * 0.1) + (index * 0.1)}s` }}
                            >
                                <CardHeader>
                                    <CardTitle className="text-lg leading-tight hover:text-primary transition-colors cursor-pointer">
                                        {post.title}
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-muted-foreground mb-4 leading-relaxed text-sm">
                                        {post.excerpt}
                                    </p>
                                    <div className="flex flex-wrap gap-1 mb-4">
                                        {post.tags.slice(0, 3).map((tag) => (
                                            <Badge key={tag} variant="secondary" className="text-xs">
                                                {tag}
                                            </Badge>
                                        ))}
                                        {post.tags.length > 3 && (
                                            <Badge variant="secondary" className="text-xs">
                                                +{post.tags.length - 3}
                                            </Badge>
                                        )}
                                    </div>
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="p-0 h-auto text-primary hover:text-primary/80"
                                        onClick={() => navigate(`/research/${post.id}`)}
                                    >
                                        Read More <ExternalLink className="ml-1 h-3 w-3" />
                                    </Button>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Research;
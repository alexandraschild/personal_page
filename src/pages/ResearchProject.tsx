import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Share2 } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import content_evalres from "@/assets/research_projects/EvalRes.md?raw";
import content_architect from "@/assets/research_projects/ai_empowered_architect.md?raw";
import content_sidewalk from "@/assets/research_projects/sidewalk_ballet.md?raw";
import { ExternalLink, Github } from "lucide-react";

interface BlogPost {
    id: string;
    title: string;
    excerpt: string;
    content: string;
    collaborators: string[];
    publications: string[];
    tags: string[];
    active: boolean;
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
        excerpt: "We explore the use of VLMs to assist architects in designing physically valid structures that align with desired emotional or aesthetic qualities. By learning disentangled representations of geometry, performance, and affective impact, we aim to bridge spatial reasoning, physical grounding, and emotional design in computational architecture.",
        collaborators: ["MIT Architecture"],
        publications: [],
        content: content_architect,
        tags: ["VLMs", "Geometry Understanding and Generation", "Affectionate Computing", "Spatial Reasoning"],
        active: true,
        links: {
            paper: "",
            code: ""
        }
    },
    {
        id: "2",
        title: "Evaluating VLMs for High-Resolution and Detail-Rich Visual Understanding",
        excerpt: "We investigate the limitations of current vision–language models (VLMs) when processing high-resolution images or small, detail-rich regions. By developing a systematic evaluation framework and novel metrics, we assess model scalability and reveal how architectural bottlenecks impact performance on fine-grained visual reasoning tasks.",
        collaborators: [],
        publications: ["Under Review"],
        content: content_evalres,
        tags: ["VLMs", "High-Resolution Vision", "Fine-Grained Visual Reasoning", "Benchmarking"],
        active: true,
        links: {
            paper: "",
            code: ""
        }
    },
    {
        id: "3",
        title: "Grounding Semantically Complex Regions in Visual Scenes",
        excerpt: "We study how vision–language models can holistically ground semantically complex and relational concepts in images. Our research aims to move beyond object-level understanding toward unified spatial, geometric, and relational reasoning over detailed real-world scenes.",
        collaborators: ["MIT City Labs"],
        publications: ["AAAI AI for Social Impact (accepted)"],
        content: content_sidewalk,
        tags: ["VLMs", "Spatial Reasoning", "Region Detection", "Semantic Grounding"],
        active: true,
        links: {
            paper: "",
            code: ""
        }
    },
    {
        id: "4",
        title: "Faithful Feature Localization in ViT using Attention-LRP",
        excerpt: "I supervised a bachelor research project investigating how Vision Transformers localize discriminative features in fine-grained re-identification. By adapting layer-wise relevance propagation to metric learning with ViTs, this work provides faithful, quantitative explanations of model decisions and formed the basis for a joint publication.",
        collaborators: ["Iven Schlegelmilch, Max Schall"],
        publications: ["WACV26 (accepted)"],
        tags: ["Vision Transformers", "Interpretability", "Layer-wise Relevance Propagation", "Fine-Grained Recognition"],
        content: "",
        active: false,
        links: {
            paper: "https://www.arxiv.org/abs/2512.07776",
            code: "https://gorilla-watch.github.io/"
        }
    },
];

const ResearchProject = () => {
    const navigate = useNavigate();
    const { id } = useParams();
    const [post, setPost] = useState<BlogPost | null>(null);

    useEffect(() => {
        const foundPost = blogPosts.find(p => p.id === id);
        setPost(foundPost || null);
    }, [id]);

    if (!post) {
        return (
            <div className="min-h-screen bg-background pt-20 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-4xl font-bold mb-4">Post Not Found</h1>
                    <p className="text-muted-foreground mb-6">The blog post you're looking for doesn't exist.</p>
                    <Button onClick={() => navigate('/research')}>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Research Projects
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background pt-20">
            <article className="container mx-auto px-6 py-12 max-w-4xl">
                {/* Navigation */}
                <Button
                    variant="ghost"
                    onClick={() => navigate('/research')}
                    className="mb-8 hover:bg-secondary/50"
                >
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Research Projects
                </Button>

                {/* Header */}
                <header className="mb-12 animate-fade-in">
                    {post.links.paper && (
                        <a
                            href={post.links.paper}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <Button variant="outline" className="hover:shadow-glow transition-all">
                                <ExternalLink className="h-5 w-5" />
                                Paper
                            </Button>

                        </a>
                    )}
                    {post.links.code && (
                        <a
                            href={post.links.paper}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <Button variant="outline" className="hover:shadow-glow transition-all">
                                <Github className="h-5 w-5" />
                                <span className="sr-only">GitHub</span>
                                Code
                            </Button>

                        </a>
                    )}

                    <h1 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
                        {post.title}
                    </h1>

                    <div className="flex items-center gap-6 text-muted-foreground mb-6">
                        <Button variant="ghost" size="sm" className="p-0 h-auto">
                            <Share2 className="h-4 w-4" />
                        </Button>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-8">
                        {post.tags.map((tag) => (
                            <Badge key={tag} variant="secondary">
                                {tag}
                            </Badge>
                        ))}
                    </div>

                    <p className="text-xl text-muted-foreground leading-relaxed">
                        {post.excerpt}
                    </p>
                </header>

                {/* Content */}
                <div className="prose prose-lg max-w-none animate-slide-up">
                    <div
                        className="text-foreground leading-relaxed space-y-6"
                        style={{
                            whiteSpace: 'pre-line',
                            fontFamily: 'inherit'
                        }}
                    >
                        {post.content.split('\n\n').map((paragraph, index) => {
                            if (paragraph.startsWith('## ')) {
                                return (
                                    <h2 key={index} className="text-2xl font-bold mt-8 mb-4 text-foreground">
                                        {paragraph.replace('## ', '')}
                                    </h2>
                                );
                            }
                            if (paragraph.startsWith('### ')) {
                                return (
                                    <h3 key={index} className="text-xl font-semibold mt-6 mb-3 text-foreground">
                                        {paragraph.replace('### ', '')}
                                    </h3>
                                );
                            }
                            if (paragraph.startsWith('```')) {
                                const lines = paragraph.split('\n');
                                const language = lines[0].replace('```', '');
                                const code = lines.slice(1, -1).join('\n');
                                return (
                                    <div key={index} className="my-6">
                                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                            <code className="text-sm text-muted-foreground">{code}</code>
                                        </pre>
                                    </div>
                                );
                            }
                            if (paragraph.startsWith('- ')) {
                                const listItems = paragraph.split('\n').filter(line => line.startsWith('- '));
                                return (
                                    <ul key={index} className="list-disc list-inside space-y-2 my-4">
                                        {listItems.map((item, itemIndex) => (
                                            <li key={itemIndex} className="text-foreground">
                                                {item.replace('- ', '')}
                                            </li>
                                        ))}
                                    </ul>
                                );
                            }
                            return paragraph.trim() && (
                                <p key={index} className="text-foreground leading-relaxed">
                                    {paragraph}
                                </p>
                            );
                        })}
                    </div>
                </div>

                {/* Footer */}
                <footer className="mt-16 pt-8 border-t border-border">
                    <div className="flex items-center justify-between">
                        <Button
                            variant="outline"
                            onClick={() => navigate('/research')}
                        >
                            <ArrowLeft className="mr-2 h-4 w-4" />
                            More Articles
                        </Button>

                        <Button variant="ghost" size="sm">
                            <Share2 className="mr-2 h-4 w-4" />
                            Share Article
                        </Button>
                    </div>
                </footer>
            </article>
        </div>
    );
};

export default ResearchProject;
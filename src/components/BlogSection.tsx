import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, Calendar, Clock } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  date: string;
  readTime: string;
  tags: string[];
  url: string;
}

const blogPosts: BlogPost[] = [
  {
    id: "1",
    title: "Building Efficient Transformer Models for Edge Devices",
    excerpt: "Exploring techniques to optimize large language models for mobile and IoT applications while maintaining performance.",
    date: "2024-01-15",
    readTime: "8 min read",
    tags: ["Transformers", "Edge AI", "Optimization"],
    url: "#"
  },
  {
    id: "2", 
    title: "The Future of Computer Vision: Beyond CNNs",
    excerpt: "Discussing emerging architectures and their impact on computer vision tasks, from Vision Transformers to Neural ODEs.",
    date: "2023-12-10",
    readTime: "12 min read", 
    tags: ["Computer Vision", "Neural Networks", "Research"],
    url: "#"
  },
  {
    id: "3",
    title: "MLOps Best Practices: From Jupyter to Production",
    excerpt: "A comprehensive guide to building robust machine learning pipelines that scale from prototype to production systems.",
    date: "2023-11-22",
    readTime: "15 min read",
    tags: ["MLOps", "DevOps", "Production ML"],
    url: "#"
  },
  {
    id: "4",
    title: "Understanding Attention Mechanisms in Modern AI",
    excerpt: "Deep dive into attention mechanisms and their evolution from seq2seq models to modern transformer architectures.",
    date: "2023-10-08", 
    readTime: "10 min read",
    tags: ["Attention", "Deep Learning", "AI Theory"],
    url: "#"
  }
];

const BlogSection = () => {
  const navigate = useNavigate();
  return (
    <section id="blog" className="py-20 bg-secondary/30">
      <div className="container mx-auto px-6">
        <div className="max-w-6xl mx-auto">
          {/* Section Header */}
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl md:text-4xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-4">
              Latest Blog Posts
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Sharing insights and discoveries from the world of AI and machine learning
            </p>
            <Button 
              variant="outline" 
              size="lg"
              onClick={() => navigate('/blog')}
              className="hover:shadow-glow transition-all"
            >
              View All Posts <ExternalLink className="ml-2 h-4 w-4" />
            </Button>
          </div>

          {/* Blog Posts Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {blogPosts.map((post, index) => (
              <Card 
                key={post.id}
                className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow animate-slide-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <CardHeader>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground mb-2">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-4 w-4" />
                      {new Date(post.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric'
                      })}
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {post.readTime}
                    </div>
                  </div>
                  <CardTitle className="text-xl leading-tight hover:text-primary transition-colors">
                    <a href={post.url} className="block">
                      {post.title}
                    </a>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4 leading-relaxed">
                    {post.excerpt}
                  </p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {post.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="p-0 h-auto text-primary hover:text-primary/80"
                    asChild
                  >
                    <a href={post.url}>
                      Read More <ExternalLink className="ml-1 h-3 w-3" />
                    </a>
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default BlogSection;
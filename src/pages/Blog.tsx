import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ExternalLink, Calendar, Clock, Search, ArrowLeft } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  date: string;
  readTime: string;
  tags: string[];
  featured: boolean;
}

const blogPosts: BlogPost[] = [
  {
    id: "1",
    title: "Title",
    excerpt: "Abstract",
    content: "Import content from md",
    date: "today",
    readTime: "x min read",
    tags: ["Transformers", "Vision Encoder", "Attention"],
    featured: false
  },
];

const Blog = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  // Get all unique tags
  const allTags = Array.from(new Set(blogPosts.flatMap(post => post.tags)));

  // Filter posts based on search and tag
  const filteredPosts = blogPosts.filter(post => {
    const matchesSearch = post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTag = !selectedTag || post.tags.includes(selectedTag);
    return matchesSearch && matchesTag;
  });

  const featuredPosts = filteredPosts.filter(post => post.featured);
  const regularPosts = filteredPosts.filter(post => !post.featured);

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

          {blogPosts.length > 1 &&
            <div>
              <div className="text-center mb-8">
                <h1 className="text-4xl md:text-5xl font-bold bg-ai-gradient bg-clip-text text-transparent mb-4">
                  AI Insights & Research
                </h1>
                <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                  Exploring the frontiers of artificial intelligence, machine learning, and emerging technologies
                </p>
              </div>

              {/* Search and Filters */}
              <div className="max-w-2xl mx-auto mb-8">
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search articles..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>

                <div className="flex flex-wrap gap-2 justify-center">
                  <Button
                    variant={selectedTag === null ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedTag(null)}
                  >
                    All Topics
                  </Button>
                  {allTags.map(tag => (
                    <Button
                      key={tag}
                      variant={selectedTag === tag ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
                    >
                      {tag}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          }
          {/* Featured Posts */}
          {featuredPosts.length > 0 && (
            <div className="mb-16">
              <h2 className="text-2xl font-bold mb-8 animate-fade-in">Featured Articles</h2>
              <div className="grid md:grid-cols-2 gap-8">
                {featuredPosts.map((post, index) => (
                  <Card
                    key={post.id}
                    className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow animate-slide-up"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <CardHeader>
                      <Badge className="w-fit mb-2 bg-primary/20 text-primary hover:bg-primary/30">
                        Featured
                      </Badge>
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
                      <CardTitle className="text-xl leading-tight hover:text-primary transition-colors cursor-pointer">
                        {post.title}
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
                        onClick={() => navigate(`/blog/${post.id}`)}
                      >
                        Read Full Article <ExternalLink className="ml-1 h-3 w-3" />
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* All Posts */}
          <div>
            <h2 className="text-2xl font-bold mb-8 animate-fade-in">
              {featuredPosts.length > 0 ? "All Articles" : ""}
            </h2>

            {filteredPosts.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No articles found matching your criteria.</p>
              </div>
            ) : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {regularPosts.map((post, index) => (
                  <Card
                    key={post.id}
                    className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/80 transition-all duration-300 hover:shadow-glow animate-slide-up"
                    style={{ animationDelay: `${(featuredPosts.length * 0.1) + (index * 0.1)}s` }}
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
                        onClick={() => navigate(`/blog/${post.id}`)}
                      >
                        Read More <ExternalLink className="ml-1 h-3 w-3" />
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Blog;
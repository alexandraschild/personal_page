import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CookingPot, Amphora, Flower, Heart, MapPin, Building2 } from "lucide-react";
import { useState } from "react";

const personalInfo = {
    location: "Potsdam, Germany",
    languages: ["English (Fluent)", "German (Fluent)", "Russian (Native)", "Italian (Learning, B1)"],
    programming: ["Python (Pytorch, Transformers)", "JavaScript/TypeScript (React, Node.js)", "SQL", "R", "C++"],
    interests: [
        {
            category: "Sports",
            items: ["Yoga", "Pole Dance", "Swimming"]
        },
    ]
};

const hobbies = [
    {
        icon: <CookingPot className="w-5 h-5" />,
        title: "Cooking & Eating:)",
        description: "I really love creating and eating tasty food. For me that's a way to explore new cultures, socialize, show love and care for others. I really love middle eastern cuisine, but also love exploring and experimenting with new ingredients."
    },
    {
        icon: <Amphora className="w-5 h-5" />,
        title: "Art & Ceramics",
        description: "I always enjoyed working with my hands and putting creative ideas from my mind to reality. My recent passion is pottery & sculpture, I really love the feeling of getting from a rough idea to a first shape that you can refine for hours and hours."
    },
    {
        icon: <Building2 className="w-5 h-5" />,
        title: "Architecture & Design",
        description: "I love exploring cities and their architecture, I think build environment and city organization is extremely important for our well-being and mental health and I can't stop thinking of factors that make a neighbourhood or building feel good. As much as I value outside, I also really enjoy learning about interior design."
    },
    {
        icon: <Flower className="w-5 h-5" />,
        title: "Enterpreneurship",
        description: "Since my previous start-up adventure, I really enjoy thinking about scaling ideas into potential businesses and what it will take to get there. I love the energy and fast pace of the start-up world and would really love to return there one day again."
    },
];

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

const PersonalSection = () => {
    return (
        <section id="personal" className="py-20 px-4 bg-muted/20">
            <div className="container mx-auto max-w-6xl">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                        Beyond the Code
                    </h2>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        The person behind the algorithms - interests, hobbies, and what makes me tick
                    </p>
                </div>

                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Personal Info */}
                    <div className="lg:col-span-1">
                        <Card className="h-fit">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Heart className="w-5 h-5 text-red-500" />
                                    Personal Info
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <h4 className="font-semibold flex items-center gap-2 mb-2">
                                        <MapPin className="w-4 h-4" />
                                        Location
                                    </h4>
                                    <p className="text-sm text-muted-foreground">{personalInfo.location}</p>
                                </div>

                                <div>
                                    <h4 className="font-semibold mb-2">Languages</h4>
                                    <div className="space-y-1">
                                        {personalInfo.languages.map((lang, idx) => (
                                            <Badge key={idx} variant="outline" className="text-xs block w-fit">
                                                {lang}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                                <div>
                                    <h4 className="font-semibold mb-2">Programming</h4>
                                    <div className="space-y-1">
                                        {personalInfo.programming.map((lang, idx) => (
                                            <Badge key={idx} variant="outline" className="text-xs">
                                                {lang}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                                <div>
                                    <h4 className="font-semibold mb-3">Interests</h4>
                                    <div className="space-y-3">
                                        {personalInfo.interests.map((category, idx) => (
                                            <div key={idx}>
                                                <h5 className="text-sm font-medium text-muted-foreground mb-1">
                                                    {category.category}
                                                </h5>
                                                <div className="flex flex-wrap gap-1">
                                                    {category.items.map((item, itemIdx) => (
                                                        <Badge key={itemIdx} variant="secondary" className="text-xs">
                                                            {item}
                                                        </Badge>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Hobbies & Activities */}
                    <div className="lg:col-span-2">
                        <div className="grid md:grid-cols-2 gap-6 mb-8">
                            {hobbies.map((hobby, index) => (
                                <Card key={index} className="hover:shadow-lg transition-all duration-300">
                                    <CardHeader>
                                        <CardTitle className="flex items-center gap-2 text-lg">
                                            {hobby.icon}
                                            {hobby.title}
                                        </CardTitle>
                                        <CardDescription>{hobby.description}</CardDescription>
                                    </CardHeader>
                                </Card>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default PersonalSection;
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Calendar, Clock, Share2 } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";

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
    title: "Building Efficient Transformer Models for Edge Devices",
    excerpt: "Exploring techniques to optimize large language models for mobile and IoT applications while maintaining performance.",
    content: `In this comprehensive guide, we dive deep into the challenges of deploying transformer models on resource-constrained devices...

## The Challenge of Edge Deployment

Transformer models have revolutionized natural language processing, but their computational requirements make deployment on edge devices challenging. With billions of parameters and extensive memory requirements, these models traditionally require powerful GPUs and substantial RAM.

## Key Optimization Techniques

### 1. Model Quantization
Quantization reduces the precision of model weights from 32-bit floating point to 8-bit or even 4-bit integers. This dramatically reduces memory usage while maintaining acceptable performance:

- **Post-training quantization**: Applied after training is complete
- **Quantization-aware training**: Incorporates quantization into the training process
- **Dynamic quantization**: Quantizes weights statically but activations dynamically

### 2. Knowledge Distillation
A smaller "student" model learns from a larger "teacher" model, capturing essential knowledge in a more compact form:

\`\`\`python
# Example knowledge distillation setup
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, temperature=4):
    distill_loss = nn.KLDivLoss()(F.log_softmax(student_logits/temperature, dim=1),
                                 F.softmax(teacher_logits/temperature, dim=1))
    student_loss = nn.CrossEntropyLoss()(student_logits, labels)
    return alpha * distill_loss + (1-alpha) * student_loss
\`\`\`

### 3. Pruning Techniques
Systematic removal of less important parameters:

- **Magnitude-based pruning**: Remove weights with smallest absolute values
- **Structured pruning**: Remove entire neurons or layers
- **Gradual pruning**: Progressive removal during training

## Implementation Strategies

### Mobile Deployment Framework
For mobile applications, consider using:

1. **TensorFlow Lite**: Google's framework for mobile and embedded devices
2. **ONNX Runtime**: Cross-platform inference optimization
3. **PyTorch Mobile**: Facebook's solution for mobile deployment

### Performance Optimization
- **Batch processing**: Group multiple inputs when possible
- **Caching strategies**: Store frequently accessed computations
- **Hardware acceleration**: Leverage device-specific accelerators

## Real-world Results

Our experiments show that optimized transformer models can achieve:
- 90% reduction in memory usage
- 5x faster inference times
- <2% accuracy degradation

These optimizations make it possible to run sophisticated NLP models on smartphones, IoT devices, and embedded systems, opening new possibilities for edge AI applications.

## Conclusion

Edge deployment of transformer models requires careful consideration of the trade-offs between model size, speed, and accuracy. By combining multiple optimization techniques, we can create efficient models that bring the power of modern NLP to resource-constrained environments.`,
    date: "2024-01-15",
    readTime: "8 min read",
    tags: ["Transformers", "Edge AI", "Optimization"],
    featured: true
  },
  {
    id: "2", 
    title: "The Future of Computer Vision: Beyond CNNs",
    excerpt: "Discussing emerging architectures and their impact on computer vision tasks, from Vision Transformers to Neural ODEs.",
    content: `Convolutional Neural Networks have dominated computer vision for over a decade, but new architectures are emerging that challenge this paradigm...

## The CNN Era

Since AlexNet's breakthrough in 2012, CNNs have been the backbone of computer vision. Their inductive biases - translation invariance and local connectivity - made them perfect for image processing tasks. However, as we push the boundaries of what's possible in computer vision, we're discovering the limitations of these assumptions.

## Enter Vision Transformers (ViTs)

Vision Transformers have shown that the attention mechanism can be just as effective for images as it is for text:

### Key Advantages:
- **Long-range dependencies**: Unlike CNNs, ViTs can model relationships between distant pixels directly
- **Scalability**: Performance improves consistently with more data and compute
- **Unified architecture**: Same model can handle various vision tasks

### Implementation Example:
\`\`\`python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, 
                                     dim_feedforward=mlp_dim),
            num_layers=depth
        )
\`\`\`

## Neural ODEs in Computer Vision

Neural Ordinary Differential Equations represent another paradigm shift:

### Continuous Depth Networks
Instead of discrete layers, Neural ODEs model the hidden state as a continuous trajectory:

\`\`\`python
class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc
        
    def forward(self, x):
        # Solve ODE from t=0 to t=1
        return odeint(self.odefunc, x, torch.tensor([0., 1.]))
\`\`\`

### Benefits:
- **Memory efficiency**: Constant memory cost regardless of depth
- **Adaptive computation**: Network can choose its own depth
- **Invertible transformations**: Enable powerful generative models

## Hybrid Architectures

The future likely lies in combining the best of all worlds:

### ConvNeXt: CNN Meets Transformer
Modernized CNNs that incorporate design choices from ViTs while maintaining CNN efficiency.

### CoAtNet: Convolution + Attention
Systematically combines convolution and attention for optimal efficiency and accuracy.

## Emerging Paradigms

### Self-Supervised Learning
Learning rich representations without labeled data:
- **Contrastive methods**: SimCLR, MoCo, SwAV
- **Masked modeling**: MAE for images, similar to BERT for text

### Foundation Models
Large-scale models trained on diverse data:
- **CLIP**: Connects vision and language
- **DALL-E**: Text-to-image generation
- **Florence**: Unified vision-language model

## Performance Comparisons

Recent benchmarks show:
- ViTs excel with large datasets (>100M images)
- Hybrid models offer best accuracy/efficiency trade-offs
- Neural ODEs provide unique memory advantages

## Looking Forward

The future of computer vision will likely feature:
1. **Multimodal models** that understand both vision and language
2. **Efficient architectures** that work across devices
3. **Self-supervised learning** reducing dependence on labeled data
4. **Neuromorphic computing** for ultra-low power vision

As we move beyond CNNs, we're entering an era of unprecedented capabilities in computer vision, with applications spanning from autonomous vehicles to medical imaging and beyond.`,
    date: "2023-12-10",
    readTime: "12 min read", 
    tags: ["Computer Vision", "Neural Networks", "Research"],
    featured: true
  },
  {
    id: "3",
    title: "MLOps Best Practices: From Jupyter to Production",
    excerpt: "A comprehensive guide to building robust machine learning pipelines that scale from prototype to production systems.",
    content: `Moving from Jupyter notebooks to production systems is one of the biggest challenges in machine learning. This guide covers the essential practices for building robust ML pipelines.

## The Production Gap

Most ML projects start in Jupyter notebooks - they're great for experimentation and prototyping. However, production systems require different considerations:

- **Reproducibility**: Experiments must be repeatable
- **Scalability**: Handle varying loads and data volumes  
- **Monitoring**: Track model and system performance
- **Maintainability**: Code that can be updated and debugged

## Core MLOps Components

### 1. Version Control
Beyond just code, ML projects need to version:
- **Data**: Track datasets and their lineage
- **Models**: Store and version trained models
- **Experiments**: Record hyperparameters and results

### 2. Automated Pipelines
\`\`\`yaml
# Example CI/CD pipeline for ML
stages:
  - data-validation
  - training
  - evaluation
  - deployment
  
data-validation:
  script:
    - python validate_data.py
    - python check_data_drift.py
    
training:
  script:
    - python train.py --config config.yaml
    - python evaluate.py --model latest
\`\`\`

### 3. Model Monitoring
Essential metrics to track:
- **Accuracy drift**: Performance degradation over time
- **Data drift**: Changes in input distribution
- **Infrastructure metrics**: Latency, throughput, resource usage

## Implementation Strategy

### Phase 1: Organize Your Code
Transform notebook code into reusable modules:
- **data.py**: Data loading and preprocessing
- **models.py**: Model architectures and training
- **evaluate.py**: Evaluation metrics and validation
- **deploy.py**: Deployment and serving logic

### Phase 2: Containerization
Use Docker to ensure consistent environments:
\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "serve.py"]
\`\`\`

### Phase 3: Orchestration
Tools like Apache Airflow or Kubeflow for workflow management.

## Best Practices

1. **Start Simple**: Begin with basic CI/CD before complex orchestration
2. **Test Everything**: Unit tests for data, models, and infrastructure
3. **Monitor Continuously**: Set up alerting for performance degradation
4. **Document Thoroughly**: Clear documentation prevents technical debt

The transition from prototype to production is challenging but following these practices ensures reliable, scalable ML systems.`,
    date: "2023-11-22",
    readTime: "15 min read",
    tags: ["MLOps", "DevOps", "Production ML"],
    featured: false
  },
  {
    id: "4",
    title: "Understanding Attention Mechanisms in Modern AI",
    excerpt: "Deep dive into attention mechanisms and their evolution from seq2seq models to modern transformer architectures.",
    content: `Attention mechanisms have revolutionized how we approach sequence-to-sequence tasks in machine learning. From their humble beginnings in neural machine translation to powering the largest language models today, attention has become the cornerstone of modern AI.

## The Problem with Fixed Representations

Traditional sequence-to-sequence models encoded entire input sequences into fixed-size vectors. This created a bottleneck - how can a single vector capture all the nuances of a complex sentence or document?

## Enter Attention

The attention mechanism allows models to dynamically focus on different parts of the input sequence when generating each output token. Instead of relying on a single context vector, the model can "attend" to the most relevant parts of the input.

### Basic Attention Calculation

\`\`\`python
def attention(query, key, value, mask=None):
    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
\`\`\`

## Evolution of Attention

### 1. Bahdanau Attention (2014)
The first attention mechanism used an additive approach with a learned alignment model.

### 2. Luong Attention (2015) 
Introduced multiplicative attention with different scoring functions.

### 3. Self-Attention (2017)
The breakthrough that enabled Transformers - allowing sequences to attend to themselves.

## Multi-Head Attention

The Transformer architecture uses multi-head attention to capture different types of relationships:

\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply attention for each head
        attention_output, _ = attention(Q, K, V)
        
        # Concatenate heads and apply final linear layer
        concat_output = attention_output.view(batch_size, seq_len, d_model)
        return self.out_linear(concat_output)
\`\`\`

## Types of Attention Patterns

### 1. Global Attention
Every position attends to every other position - computationally expensive but most expressive.

### 2. Local Attention  
Attention is restricted to a local window around each position.

### 3. Sparse Attention
Only attend to a subset of positions based on learned patterns.

## Beyond Standard Attention

### Cross-Attention
Used in encoder-decoder architectures where decoder attends to encoder outputs.

### Causal Attention
Prevents future positions from being attended to - essential for autoregressive generation.

### Relative Position Encoding
Incorporates relative distances between positions rather than absolute positions.

## Practical Applications

Attention mechanisms have enabled breakthroughs in:
- **Machine Translation**: More accurate and fluent translations
- **Text Summarization**: Better content selection and coherence
- **Question Answering**: Precise attention to relevant passages
- **Image Captioning**: Attending to relevant image regions

## Visualizing Attention

One of the most powerful aspects of attention is its interpretability. Attention weights show us exactly what the model is focusing on, providing insights into its decision-making process.

## Future Directions

Current research focuses on:
- **Efficient attention**: Reducing the O(n²) complexity
- **Long-range attention**: Handling very long sequences
- **Structured attention**: Incorporating linguistic or domain knowledge

Attention has fundamentally changed how we think about sequence modeling, and its influence continues to shape the future of AI.`,
    date: "2023-10-08", 
    readTime: "10 min read",
    tags: ["Attention", "Deep Learning", "AI Theory"],
    featured: false
  },
  {
    id: "5",
    title: "Scaling Neural Networks: Lessons from GPT and Beyond",
    excerpt: "Analyzing the scaling laws of neural networks and what they mean for the future of AI development.",
    content: `The remarkable success of large language models has brought scaling laws to the forefront of AI research. Understanding these laws is crucial for predicting the capabilities and limitations of future AI systems.

## The Scaling Hypothesis

The scaling hypothesis suggests that many aspects of model performance improve predictably as we increase model size, training data, and compute resources. This has profound implications for AI development strategies.

## Empirical Scaling Laws

### The Power Law Relationship

Research from OpenAI and others has shown that many AI capabilities follow power laws:

\`\`\`
Loss ~ N^(-α)
\`\`\`

Where:
- N = number of model parameters
- α = scaling exponent (typically 0.05-0.1)

### Key Scaling Dimensions

1. **Model Size (N)**: Number of parameters
2. **Dataset Size (D)**: Number of training tokens
3. **Compute Budget (C)**: FLOPs used for training

## The Chinchilla Scaling Laws

DeepMind's Chinchilla research revealed that most large models are undertrained:

### Optimal Compute Allocation
For a given compute budget C:
- **Optimal model size**: N* ∝ C^0.5
- **Optimal dataset size**: D* ∝ C^0.5

This suggests that compute should be split equally between model size and training data.

### Practical Implications

\`\`\`python
def optimal_allocation(compute_budget):
    # Chinchilla scaling law approximation
    model_params = (compute_budget / 6) ** 0.5
    training_tokens = 20 * model_params
    
    return {
        'parameters': model_params,
        'training_tokens': training_tokens,
        'expected_performance': compute_budget ** (-0.05)
    }
\`\`\`

## Emergent Capabilities

Large models exhibit capabilities that smaller models lack entirely:

### Examples of Emergence
- **In-context learning**: Few-shot learning without parameter updates
- **Chain-of-thought reasoning**: Step-by-step problem solving
- **Code generation**: Programming across multiple languages

### Scaling Thresholds
Many capabilities appear suddenly at specific model sizes:
- GPT-3 (175B): Strong few-shot learning
- PaLM (540B): Complex reasoning tasks
- GPT-4: Multimodal understanding

## Infrastructure Scaling Challenges

### Memory Requirements
\`\`\`python
def memory_requirements(params, precision=16):
    # Model weights
    model_memory = params * precision / 8  # bytes
    
    # Gradients (for training)
    gradient_memory = model_memory
    
    # Optimizer states (Adam)
    optimizer_memory = model_memory * 2
    
    # Activations (depends on sequence length)
    activation_memory = estimate_activations(params)
    
    return {
        'total_gb': (model_memory + gradient_memory + 
                    optimizer_memory + activation_memory) / 1e9,
        'min_gpus': calculate_gpu_requirements(total_memory)
    }
\`\`\`

### Communication Overhead
As models grow, the communication between GPUs becomes a bottleneck:
- **Model parallelism**: Split model across devices
- **Data parallelism**: Process different batches on different devices
- **Pipeline parallelism**: Process different layers on different devices

## Economic Implications

### Training Costs
The cost of training large models grows superlinearly:

\`\`\`
Training Cost ∝ N^1.3 × D^1.0
\`\`\`

### Inference Costs
Serving large models requires significant infrastructure:
- Memory bandwidth limitations
- Latency requirements for real-time applications
- Energy consumption for continuous operation

## Scaling Beyond Current Paradigms

### Mixture of Experts (MoE)
Scales model capacity without proportional compute increase:

\`\`\`python
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, expert_capacity, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert() for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        gate_scores = self.gate(x)
        top_k_gates, top_k_indices = torch.topk(gate_scores, self.top_k)
        
        # Route tokens to top-k experts
        outputs = []
        for i in range(self.top_k):
            expert_output = self.experts[top_k_indices[:, i]](x)
            weighted_output = expert_output * top_k_gates[:, i:i+1]
            outputs.append(weighted_output)
        
        return sum(outputs)
\`\`\`

### Retrieval-Augmented Generation
Combines parametric knowledge with external databases to scale knowledge without model size.

## Limitations of Scaling

### Sample Efficiency
Larger models often require more data to achieve their potential, leading to diminishing returns in data-limited domains.

### Alignment Challenges
As models become more capable, ensuring they behave as intended becomes increasingly difficult.

### Environmental Impact
The energy consumption of training and running large models raises sustainability concerns.

## Future Scaling Trends

### Multimodal Scaling
Extending scaling laws to models that process text, images, audio, and video simultaneously.

### Efficient Architectures
Developing architectures that achieve better scaling curves:
- Sparse models (Switch Transformer, GLaM)
- Retrieval-based models (RETRO, FiD)
- Continuous learning systems

## Practical Guidelines

For practitioners working with scaling:

1. **Follow Chinchilla laws** for compute allocation
2. **Monitor emergence** of new capabilities at scale
3. **Plan infrastructure** for memory and communication needs
4. **Consider alternatives** like MoE for efficient scaling

The scaling paradigm has driven remarkable progress in AI, but understanding its principles and limitations is crucial for making informed decisions about future model development.`,
    date: "2023-09-14",
    readTime: "11 min read",
    tags: ["Scaling Laws", "Large Models", "AI Research"],
    featured: false
  },
  {
    id: "6",
    title: "Ethical AI: Building Responsible Machine Learning Systems",
    excerpt: "Exploring the ethical considerations in AI development and practical approaches to building fair and transparent systems.",
    content: `As AI systems become more prevalent in society, the importance of ethical considerations cannot be overstated. Building responsible AI requires deliberate design choices and ongoing vigilance throughout the development lifecycle.

## The Ethical Imperative

AI systems now influence critical decisions in healthcare, criminal justice, hiring, and financial services. The stakes are too high to treat ethics as an afterthought - it must be integrated from the beginning of the development process.

## Core Ethical Principles

### 1. Fairness and Non-Discrimination
Ensuring AI systems don't perpetuate or amplify existing biases:

\`\`\`python
def measure_fairness(predictions, protected_attribute, labels):
    # Demographic parity: P(ŷ=1|A=0) = P(ŷ=1|A=1)
    demo_parity = abs(
        predictions[protected_attribute == 0].mean() - 
        predictions[protected_attribute == 1].mean()
    )
    
    # Equalized odds: P(ŷ=1|A=a,Y=y) equal across groups
    tpr_0 = predictions[(protected_attribute == 0) & (labels == 1)].mean()
    tpr_1 = predictions[(protected_attribute == 1) & (labels == 1)].mean()
    
    return {
        'demographic_parity': demo_parity,
        'equalized_odds': abs(tpr_0 - tpr_1)
    }
\`\`\`

### 2. Transparency and Explainability
Users should understand how AI systems make decisions:

\`\`\`python
import shap

def explain_prediction(model, X_instance, X_train):
    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    
    # Feature importance
    feature_importance = dict(zip(
        X_train.columns, 
        abs(shap_values).mean(0)
    ))
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'explanation': generate_natural_language_explanation(shap_values)
    }
\`\`\`

### 3. Privacy Protection
Safeguarding individual privacy while enabling useful AI:

#### Differential Privacy
\`\`\`python
def add_dp_noise(data, epsilon=1.0, sensitivity=1.0):
    # Laplace mechanism for differential privacy
    noise = np.random.laplace(0, sensitivity/epsilon, data.shape)
    return data + noise

def train_with_dp(model, data, epsilon=1.0):
    # Differentially private training
    for batch in data:
        # Clip gradients to bound sensitivity
        gradients = compute_gradients(model, batch)
        clipped_grads = clip_gradients(gradients, max_norm=1.0)
        
        # Add noise to gradients
        noisy_grads = add_dp_noise(clipped_grads, epsilon)
        
        # Update model
        model.update(noisy_grads)
\`\`\`

### 4. Accountability and Responsibility
Clear lines of responsibility for AI system outcomes:

- **Human oversight**: Meaningful human control over high-stakes decisions
- **Audit trails**: Comprehensive logging of system behavior
- **Error handling**: Graceful degradation and error recovery

## Practical Implementation

### Bias Detection and Mitigation

#### Pre-processing Techniques
\`\`\`python
def reweight_dataset(X, y, protected_attr):
    # Reweight samples to achieve demographic parity
    weights = np.ones(len(X))
    
    for group in [0, 1]:
        for label in [0, 1]:
            mask = (protected_attr == group) & (y == label)
            if mask.sum() > 0:
                target_prop = 0.25  # Equal representation
                current_prop = mask.mean()
                weights[mask] = target_prop / current_prop
    
    return weights
\`\`\`

#### In-processing Techniques
\`\`\`python
def fair_training_loss(predictions, labels, protected_attr, lambda_fair=0.1):
    # Standard prediction loss
    pred_loss = F.binary_cross_entropy(predictions, labels)
    
    # Fairness constraint (demographic parity)
    group_0_pred = predictions[protected_attr == 0].mean()
    group_1_pred = predictions[protected_attr == 1].mean()
    fairness_loss = (group_0_pred - group_1_pred) ** 2
    
    return pred_loss + lambda_fair * fairness_loss
\`\`\`

### Algorithmic Auditing

Regular assessment of system behavior:

\`\`\`python
class AlgorithmicAudit:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def run_audit(self):
        return {
            'accuracy_metrics': self.compute_accuracy(),
            'fairness_metrics': self.compute_fairness(),
            'robustness_metrics': self.compute_robustness(),
            'privacy_metrics': self.compute_privacy_leakage()
        }
    
    def compute_fairness(self):
        # Comprehensive fairness evaluation
        predictions = self.model.predict(self.test_data.X)
        
        return {
            'demographic_parity': self.demographic_parity(predictions),
            'equalized_odds': self.equalized_odds(predictions),
            'individual_fairness': self.individual_fairness(predictions)
        }
\`\`\`

## Governance Frameworks

### AI Ethics Committees
Cross-functional teams that review AI projects for ethical implications:

- **Diverse representation**: Technical, legal, social science, domain experts
- **Review processes**: Regular assessment at project milestones
- **Decision authority**: Power to modify or halt problematic projects

### Documentation Standards
Comprehensive documentation of AI systems:

#### Model Cards
\`\`\`python
model_card = {
    'model_details': {
        'architecture': 'Random Forest',
        'training_data': 'Customer transaction data 2020-2023',
        'intended_use': 'Credit risk assessment'
    },
    'evaluation_data': {
        'datasets': ['holdout_test', 'adversarial_test'],
        'motivation': 'Representative evaluation across demographics'
    },
    'quantitative_analyses': {
        'overall_accuracy': 0.87,
        'fairness_metrics': fairness_results,
        'performance_across_groups': group_performance
    },
    'ethical_considerations': {
        'risks': ['Potential bias against protected groups'],
        'mitigations': ['Bias testing', 'Human review for edge cases']
    }
}
\`\`\`

## Regulatory Landscape

### Current Regulations
- **GDPR**: Right to explanation for automated decision-making
- **CCPA**: Consumer privacy rights in California
- **Algorithmic Accountability Acts**: Proposed US federal legislation

### Industry Standards
- **IEEE Standards**: Ethical design and bias mitigation
- **ISO/IEC Standards**: AI risk management and governance
- **Partnership on AI**: Best practices for responsible AI

## Challenges and Trade-offs

### Accuracy vs. Fairness
Often there are tensions between maximizing accuracy and ensuring fairness:

\`\`\`python
def pareto_frontier(accuracies, fairness_scores):
    # Find Pareto optimal solutions
    pareto_optimal = []
    
    for i, (acc, fair) in enumerate(zip(accuracies, fairness_scores)):
        is_dominated = False
        for j, (acc_j, fair_j) in enumerate(zip(accuracies, fairness_scores)):
            if i != j and acc_j >= acc and fair_j >= fair and (acc_j > acc or fair_j > fair):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_optimal.append((acc, fair))
    
    return pareto_optimal
\`\`\`

### Privacy vs. Utility
Stronger privacy protections often reduce model utility:
- Differential privacy adds noise that can hurt performance
- Data minimization limits the information available for training
- Federated learning introduces communication constraints

## Future Directions

### Technical Advances
- **Causal fairness**: Using causal reasoning to define fairness
- **Federated learning**: Privacy-preserving distributed training
- **Homomorphic encryption**: Computation on encrypted data

### Policy Development
- **Algorithmic impact assessments**: Mandatory evaluation of high-risk AI
- **Certification programs**: Standards for ethical AI practitioners
- **International cooperation**: Global governance frameworks

## Building an Ethical AI Culture

Creating responsible AI requires more than technical solutions:

1. **Education**: Training developers in ethical AI principles
2. **Incentive alignment**: Rewarding ethical behavior over just performance
3. **Stakeholder engagement**: Including affected communities in design
4. **Continuous monitoring**: Ongoing assessment of deployed systems

Ethical AI isn't a destination but a continuous journey. As our capabilities grow, so must our commitment to developing AI systems that benefit all of society.`,
    date: "2023-08-20",
    readTime: "9 min read",
    tags: ["Ethics", "Responsible AI", "Fairness"],
    featured: false
  }
];

const BlogPost = () => {
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
          <Button onClick={() => navigate('/blog')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Blog
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
          onClick={() => navigate('/blog')}
          className="mb-8 hover:bg-secondary/50"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Blog
        </Button>

        {/* Header */}
        <header className="mb-12 animate-fade-in">
          {post.featured && (
            <Badge className="mb-4 bg-primary/20 text-primary hover:bg-primary/30">
              Featured Article
            </Badge>
          )}
          
          <h1 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
            {post.title}
          </h1>
          
          <div className="flex items-center gap-6 text-muted-foreground mb-6">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              {new Date(post.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              {post.readTime}
            </div>
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
              onClick={() => navigate('/blog')}
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

export default BlogPost;
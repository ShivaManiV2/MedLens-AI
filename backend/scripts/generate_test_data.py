from fpdf import FPDF

docs = [
    {
        "title": "Vitamin D Supplementation and Bone Density: A Decade-Long Trial",
        "authors": "Dr. Sarah Jenkins, Dr. Michael Wong",
        "year": "2024",
        "journal": "Journal of Clinical Endocrinology",
        "content": """Abstract
Vitamin D is a crucial nutrient for bone health. We studied the long-term effects of daily Vitamin D supplementation (2000 IU) on postmenopausal women over a 10-year period.

Introduction
Bone density loss is a common issue. Vitamin D facilitates calcium absorption, making it vital. However, the optimal dosage remains debated in recent years. We investigate the impact of a moderate daily dose.

Methods
A randomized, double-blind, placebo-controlled trial was conducted with 5000 participants. The treatment group received 2000 IU daily. Bone mineral density (BMD) was measured annually at the lumbar spine and femoral neck.

Results
At Year 10, the treatment group showed a 4.2% increase in lumbar spine BMD compared to a 1.5% decrease in the placebo group. The fracture rate was significantly lower in the treatment group.

Conclusion
Daily supplementation with 2000 IU of Vitamin D is safe and effectively preserves bone density, reducing fracture risk over the long term."""
    },
    {
        "title": "Efficacy of Novel mRNA Vaccines in Autoimmune Disorders",
        "authors": "Dr. Elena Rodriguez",
        "year": "2025",
        "journal": "Immunology Today",
        "content": """Abstract
The rapid development of mRNA vaccines has revolutionized preventative medicine. We evaluate the safety and efficacy of these vaccines in patients with preexisting autoimmune conditions like Lupus and Rheumatoid Arthritis.

Introduction
Patients with autoimmune conditions are often excluded from early clinical trials. The mechanisms of mRNA vaccines have raised theoretical concerns about exacerbating autoimmune flares. This study addresses this gap.

Methods
We conducted a prospective observational cohort study of 2000 patients with autoimmune diseases who received a standard dose of the mRNA-1273 vaccine. Flare rates were monitored for 6 months post-vaccination.

Results
The incidence of disease flare was 4% in vaccinated patients, which is statistically indistinguishable from the background flare rate of the unvaccinated cohort during the same period.

Conclusion
The mRNA vaccines do not significantly increase the risk of disease flares in patients with stable autoimmune disorders. Vaccination should be strongly considered for this vulnerable demographic."""
    },
    {
        "title": "Dietary Fiber and its Role in Gut Microbiome Diversity",
        "authors": "Dr. Hans Mueller",
        "year": "2023",
        "journal": "Gastroenterology Research",
        "content": """Abstract
Dietary fiber is essential for maintaining a healthy gut microbiome. Our study analyzes how different types of fiber influence microbial diversity and short-chain fatty acid (SCFA) production.

Introduction
The human digestive system lacks the enzymes required to break down most fiber. Instead, these complex carbohydrates are fermented by gut bacteria, producing SCFAs like butyrate, which have systemic anti-inflammatory effects.

Methods
150 participants were placed on randomized diets consisting of varying ratios of soluble and insoluble fiber. Fecal samples were sequenced sequentially over 12 weeks.

Results
Participants on high-soluble fiber diets showed a 30% relative increase in Butyricicoccus levels. SCFAs measured in serum also increased proportionally with fiber intake.

Conclusion
Increasing dietary intake of soluble fiber drastically improves gut microbiome diversity and increases production of beneficial organic acids."""
    }
]

for i, doc in enumerate(docs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=doc['title'], ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, txt=f"Authors: {doc['authors']} | Year: {doc['year']} | Journal: {doc['journal']}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    # Split content by newlines and handle each line
    for line in doc['content'].split('\n'):
        # Just simple encoding hack to avoid character issues
        encoded_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=encoded_line)

    filename = f"backend/data/doc_{i+1}.pdf"
    pdf.output(filename)
    print(f"Generated {filename}")

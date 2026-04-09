import json
import csv
import random

# -----------------------------------------------------
# 1. Generate Senior Interview Experiences JSON
# -----------------------------------------------------

companies = ["Google", "Amazon", "Microsoft", "Goldman Sachs", "Atlassian", "Uber", "Tower Research", "Sprinklr", "Morgan Stanley"]
outcomes = ["Selected", "Selected", "Selected", "Waitlisted", "Rejected"]

tech_stacks = [
    ["Python", "Django", "Machine Learning"],
    ["Java", "Spring Boot", "AWS", "System Design"],
    ["C++", "DSA", "System Design", "OS"],
    ["JavaScript", "React", "Node.js", "MongoDB"],
    ["Go", "Kubernetes", "Docker", "Microservices"],
    ["Python", "Data Science", "SQL", "Tableau"],
    ["C++", "Quantitative Finance", "Low Latency"],
    ["Java", "Android", "Kotlin", "Firebase"]
]

first_names = ["Arjun", "Aditi", "Rahul", "Priya", "Karan", "Neha", "Rohit", "Sneha", "Vikram", "Anjali"]
last_names = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Kumar", "Mehta", "Jain", "Desai", "Joshi"]

experiences = []

for i in range(1, 41):
    first = random.choice(first_names)
    last = random.choice(last_names)
    company = random.choice(companies)
    stack = random.choice(tech_stacks)
    
    xp_text = f"My interview experience at {company} was challenging but rewarding. The first round was heavily focused on DSA and problem-solving, mostly array and graph questions. Since my tech stack includes {', '.join(stack)}, they asked me deep design questions related to that in the second round. I had to design a scalable concurrent system. The behavioral round focused on my {random.randint(2, 5)} projects. Strong communication really helped me explain my thought process clearly."
    
    exp = {
        "id": i,
        "student_name": f"{first} {last}",
        "company": company,
        "cgpa": round(random.uniform(7.5, 9.8), 2),
        "tech_stack": stack,
        "projects": random.randint(1, 5),
        "internships": random.randint(0, 3),
        "open_source": random.choice([True, False]),
        "experience_text": xp_text,
        "outcome": random.choice(outcomes)
    }
    experiences.append(exp)

with open("./data/senior_experiences.json", "w") as f:
    json.dump(experiences, f, indent=4)
print("✅ Generated data/senior_experiences.json (40 records)")

# -----------------------------------------------------
# 2. Generate Placement Training Data CSV
# -----------------------------------------------------
# We will create a synthetic dataset of 500 rows where:
# score = function(cgpa, projects, internships, comm_score, os_score, tech_score) + noise
import math

csv_data = [["cgpa", "tech_stack_score", "num_projects", "num_internships", "communication_score", "open_source_score", "readiness_score"]]

for _ in range(500):
    cgpa = round(random.uniform(6.0, 9.9), 2)
    tech_score = random.randint(2, 10) # Out of 10
    projects = random.randint(0, 6)
    internships = random.randint(0, 3)
    comm_score = random.randint(1, 5)
    os_score = random.randint(0, 2)
    
    # Heuristic scoring function
    # Max possible base: (10*4) + (10*2) + (6*2) + (3*4) + (5*2) + (2*4) 
    # Let's weigh them nicely out of 100
    # CGPA: max 40
    # Tech: max 20
    # Projects: max 15 (2.5 each)
    # Internships: max 12 (4 each)
    # Comm: max 8 (1.6 each)
    # OS: max 5 (2.5 each)
    
    score = (cgpa / 10.0 * 40) + \
            (tech_score / 10.0 * 20) + \
            (projects * 2.5) + \
            (internships * 4.0) + \
            (comm_score * 1.6) + \
            (os_score * 2.5)
            
    # Cap and add noise
    noise = random.uniform(-4, 4)
    final_score = min(max(int(score + noise), 0), 100)
    
    csv_data.append([cgpa, tech_score, projects, internships, comm_score, os_score, final_score])

with open("./data/placement_training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
print("✅ Generated data/placement_training_data.csv (500 records)")
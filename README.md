# PAARAL: Executive Presentation Mockup

This is an interactive, frontend-only mockup of **PAARAL** (Platform for Analyzing Access and Resource Allocation in Learning). It is designed specifically for executive demonstrations (DepEd ExeCom, Secretary-level briefings) by the Education Center for AI Research (ECAIR).

## Project Context
PAARAL addresses **information asymmetry** in the Philippine basic education system. When Grade 6 students finish elementary school, their households face a critical decision: stay at the current public school, transfer, or apply for an **ESC (Educational Service Contracting)** slot at a private school. Currently, the ESC option is largely invisible to many families.

This branch contains the **Student View**: a unified platform showing a family their complete set of Junior High School options—public and private—with ESC subsidies applied, live slot availability, net costs, and commute estimates.

## Tech Stack
* **Framework:** React + Vite
* **Styling:** Tailwind CSS v4
* **Icons:** Lucide React
* **Map Engine:** Custom SVG projection (No external map library dependencies)

## Running Locally
To spin this up on your local machine:

1. **Clone this specific branch:**
   ```bash
   git clone -b mockups/student-view [https://github.com/cair-philippines/project_paaral.git](https://github.com/cair-philippines/project_paaral.git)
   ```
2. **Navigate to the directory:**
   ```bash
   cd project_paaral
   ```
3. **Install dependencies:**
   ```bash
   npm install
   ```
4. **Start the development server:**
   ```bash
   npm run dev
   ```
5. **View the app:**
   Open your browser to the local address provided in your terminal (usually `http://localhost:5173`).

## Deployment
This app is optimized for zero-config deployment on **Vercel** or **Netlify**. 

When connecting this repository to a deployment service, ensure the **Production Branch** is set specifically to `mockups/student-view` to ensure the user views the interactive frontend mockup rather than the repository's main analysis code.
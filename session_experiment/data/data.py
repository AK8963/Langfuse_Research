from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()
langfuse = get_client()

# Create the dataset
dataset = langfuse.create_dataset(
    name="6-turn-travel-conversations",
    description="6-turn conversations for European vacation recommendations"
)

# Add dataset items (test cases) - expanded to 10 items
langfuse.create_dataset_item(
    dataset_name="6-turn-travel-conversations",
    input={
        "persona": "A solo traveler in their 30s looking for adventure and nature.",
        "scenario": "I want to plan a vacation in Europe. Can you help me find good places?"
    },
    expected_output="Specific mountain destinations in Europe with hiking trails suitable for solo travelers"
)

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A couple looking for a romantic getaway with scenic views.",
#         "scenario": "We're planning our anniversary trip to Europe. What do you recommend?"
#     },
#     expected_output="Romantic mountain destinations in Europe with scenic views and hiking"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A family with teenagers who enjoy outdoor activities.",
#         "scenario": "We want a European vacation that everyone will enjoy. Any suggestions?"
#     },
#     expected_output="Family-friendly mountain destinations with varied hiking trails and activities"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A budget-conscious backpacker seeking authentic experiences.",
#         "scenario": "I'm backpacking through Europe on a tight budget. Where should I go?"
#     },
#     expected_output="Affordable mountain destinations in Europe with hostels and free hiking trails"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A retired couple looking for peaceful, less crowded destinations.",
#         "scenario": "We want a quiet European getaway away from tourist crowds. Any ideas?"
#     },
#     expected_output="Peaceful, off-the-beaten-path mountain destinations with moderate hiking"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A photographer seeking dramatic landscapes and unique vistas.",
#         "scenario": "I'm looking for photogenic mountain locations in Europe. Where should I go?"
#     },
#     expected_output="Scenic mountain destinations in Europe with dramatic landscapes and photography opportunities"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A wellness enthusiast looking for relaxation and nature therapy.",
#         "scenario": "I need a restorative mountain retreat in Europe. Any recommendations?"
#     },
#     expected_output="Wellness-focused mountain destinations with spa facilities and gentle nature trails"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "An experienced mountaineer seeking challenging climbs.",
#         "scenario": "I want to tackle some serious peaks in Europe. What are my options?"
#     },
#     expected_output="Challenging mountain destinations in Europe with technical climbing routes"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A food lover interested in local cuisine and mountain villages.",
#         "scenario": "I want to explore European mountain regions with great local food. Suggestions?"
#     },
#     expected_output="Mountain destinations in Europe known for traditional cuisine and charming villages"
# )

# langfuse.create_dataset_item(
#     dataset_name="6-turn-travel-conversations",
#     input={
#         "persona": "A winter sports enthusiast planning a ski trip.",
#         "scenario": "I'm looking for the best ski resorts in the European mountains. Where should I go?"
#     },
#     expected_output="Top European mountain ski resorts with varied terrain and facilities"
# )

# print("✅ Dataset created successfully!")
# print(f"Dataset name: 6-turn-travel-conversations")
# print(f"Number of items: 10")
from pydantic import BaseModel, Field

class UpsertItem(BaseModel):
    post_id: int = Field(0, example=1)
    text: str = Field("", example="Check out my new post!")
    post_length: int = Field(0, example=120)
    num_images: int = Field(0, example=2)
    num_videos: int = Field(0, example=0)
    num_hashtags: int = Field(0, example=3)
    author_followers: int = Field(0, example=1500)
    author_following: int = Field(0, example=300)
    author_posts_last_week: int = Field(0, example=2)
    post_type: str = Field("text", example="image")  # default 'text'
    post_time_hour: int = Field(0, example=14)
    is_boosted: int = Field(0, example=1)

class QueryRequest(BaseModel):
    age: int = Field(0, example=25)
    gender: int = Field(0, example=1)
    num_friends: int = Field(0, example=100)
    avg_likes_received: float = Field(0.0, example=12.5)
    avg_comments_received: float = Field(0.0, example=3.2)
    avg_shares_received: float = Field(0.0, example=1.5)
    active_days_last_week: int = Field(0, example=5)
    time_spent_last_week: float = Field(0.0, example=120.0)
    num_groups: int = Field(0, example=4)
    has_profile_picture: int = Field(0, example=1)
    top_k: int = Field(10, example=5)
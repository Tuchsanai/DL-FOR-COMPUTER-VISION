# Set Git user configuration using variables
# Define GitHub repository URL and Personal Access Token

user_name=""
user_email=""
GITHUB_TOKEN=""
REPO_URL=""




git config --global user.name "$user_name"
git config --global user.email "$user_email"


# Format the URL with the token for authentication
AUTHENTICATED_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"

# Extract repository directory name from REPO_URL
REPO_DIR=$(basename -s .git $REPO_URL)

# Check if the folder does not exist, then clone the repository
if [ ! -d "$REPO_DIR" ]; then
    git clone $AUTHENTICATED_URL
else
    # If the folder exists, navigate to it and perform git operations
    cd "$REPO_DIR"
    
    # Pull the latest changes from the repository
    git pull $AUTHENTICATED_URL main
    
    # Generate a random commit message
    RANDOM_MESSAGE="Auto-commit $(date +%Y-%m-%d:%H:%M:%S)"
    
    # Stage changes, commit with a random message, and push to the repository
    git add .
    git commit -m "$RANDOM_MESSAGE"
    git push $AUTHENTICATED_URL main
    
    # Return to the original directory
    cd ..
fi

# Check if VS Code is installed
if command -v code >/dev/null 2>&1; then
    echo "VS Code is installed. Checking extensions..."

    # Function to check and install an extension
    install_extension() {
        local extension=$1
        if code --list-extensions | grep -q "^${extension}$"; then
            echo "Extension '${extension}' is already installed."
        else
            echo "Installing extension '${extension}'..."
            code --install-extension "${extension}"
        fi
    }

    # List of extensions to install
    extensions=(
        "ms-python.python"          # Python extension
        "ms-toolsai.jupyter"        # Jupyter extension
        "GitHub.copilot"            # GitHub Copilot
        "ms-azuretools.vscode-docker" # Docker extension
      
    )

    # Loop through and check/install each extension
    for extension in "${extensions[@]}"; do
        install_extension "${extension}"
    done

    echo "Extension check and installation complete."
else
    echo "VS Code is not installed. Please install VS Code to use this script."
fi

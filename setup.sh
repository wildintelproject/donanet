#!/usr/bin/env bash
# ============================================================================ #
#                    DonaNet — Environment Setup Script
# ============================================================================ #
# Sets up a Python virtual environment with pip and installs DonaNet
# in editable mode, following the installation workflow described in README.md.
#
# Usage:
#   ./setup.sh
# ============================================================================ #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                  DonaNet — Environment Setup                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

wait_before_exit() {
    echo ""
    read -r -p "Press Enter to close..."
}

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python was not found. Please install Python 3.11 or newer."
        wait_before_exit
        exit 1
    fi

    print_success "Python found: $($PYTHON_CMD --version)"
}

create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment already exists — reusing ${VENV_DIR}"
        return 0
    fi

    print_info "Creating virtual environment in ${VENV_DIR}..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    print_success "Virtual environment created"
}

activate_venv() {
    print_info "Activating virtual environment..."

    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # Linux/macOS
        source "$VENV_DIR/bin/activate"
    elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        # Windows Git Bash
        source "$VENV_DIR/Scripts/activate"
    else
        print_error "Could not find the virtual environment activation script."
        print_error "Checked:"
        print_error "${VENV_DIR}/bin/activate"
        print_error "${VENV_DIR}/Scripts/activate"
        wait_before_exit
        exit 1
    fi

    print_success "Virtual environment activated"
}

install_dependencies() {
    print_info "Upgrading pip..."
    python -m pip install --upgrade pip

    print_info "Installing DonaNet and required dependencies with pip..."
    pip install -e .

    print_success "Dependencies installed"
}

install_docs_dependencies_optional() {
    echo ""
    read -r -p "Install documentation dependencies as well? [y/N]: " INSTALL_DOCS

    case "$INSTALL_DOCS" in
        [yY][eE][sS]|[yY])
            print_info "Installing documentation dependencies..."
            pip install -e ".[docs]"
            print_success "Documentation dependencies installed"
            ;;
        *)
            print_info "Skipping documentation dependencies"
            ;;
    esac
}

print_next_steps() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                       Setup complete!                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    print_info "Activate the environment before running DonaNet commands:"
    echo ""
    echo "  Linux/macOS:"
    echo "    source .venv/bin/activate"
    echo ""
    echo "  Windows Git Bash:"
    echo "    source .venv/Scripts/activate"
    echo ""
    echo "  Windows PowerShell:"
    echo "    .venv\\Scripts\\Activate.ps1"
    echo ""
    echo "  Windows Command Prompt:"
    echo "    .venv\\Scripts\\activate"
    echo ""

    print_info "Useful DonaNet commands:"
    echo ""
    echo "  python donanet.py info"
    echo "  python donanet.py list-datasets"
    echo "  python donanet.py test --weights weights/donanet_weights.pt --conf 0.25"
    echo "  python donanet.py train"
    echo ""

    print_warning "For training, make sure that dataset/data.yaml exists and points to the correct dataset paths."
    print_warning "For full evaluation statistics, dataset/annotations.csv must exist and follow the expected format."
    echo ""
}

main() {
    print_header
    cd "$SCRIPT_DIR"

    check_python
    create_venv
    activate_venv
    install_dependencies
    install_docs_dependencies_optional
    print_next_steps
    wait_before_exit
}

main
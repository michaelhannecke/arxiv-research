class ArxivProcessor {
    constructor() {
        this.apiBase = '/api';
        this.currentTaskId = null;
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.pollingInterval = null;
        this.historyData = [];
        this.filteredHistory = [];
        
        this.init();
    }
    
    init() {
        this.initTheme();
        this.bindEvents();
        this.loadHistory();
        this.checkAIStatus();
    }
    
    initTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        const themeIcon = document.querySelector('#toggleTheme i');
        if (themeIcon) {
            themeIcon.className = this.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
    
    bindEvents() {
        // Theme toggle
        document.getElementById('toggleTheme')?.addEventListener('click', () => {
            this.toggleTheme();
        });
        
        // Process paper
        document.getElementById('processBtn')?.addEventListener('click', () => {
            this.processNewPaper();
        });
        
        // Cancel processing
        document.getElementById('cancelBtn')?.addEventListener('click', () => {
            this.cancelProcessing();
        });
        
        // Enter key for input
        document.getElementById('arxivInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processNewPaper();
            }
        });
        
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.switchTab(btn.dataset.tab);
            });
        });
        
        // History search
        document.getElementById('historySearch')?.addEventListener('input', (e) => {
            this.filterHistory(e.target.value);
        });
        
        // Refresh history
        document.getElementById('refreshHistory')?.addEventListener('click', () => {
            this.loadHistory();
        });
        
        // Modal close
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => {
                this.closeModal();
            });
        });
        
        // Click outside modal to close
        document.getElementById('documentModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'documentModal') {
                this.closeModal();
            }
        });
        
        // Export document
        document.getElementById('exportBtn')?.addEventListener('click', () => {
            this.exportDocument();
        });
    }
    
    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
        
        const themeIcon = document.querySelector('#toggleTheme i');
        if (themeIcon) {
            themeIcon.className = this.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');
    }
    
    async processNewPaper() {
        const arxivInput = document.getElementById('arxivInput');
        const processBtn = document.getElementById('processBtn');
        const arxivId = arxivInput.value.trim();
        
        if (!arxivId) {
            this.showToast('Please enter an arXiv ID', 'warning');
            return;
        }
        
        if (!this.validateArxivId(arxivId)) {
            this.showToast('Invalid arXiv ID format', 'error');
            return;
        }
        
        try {
            processBtn.disabled = true;
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            const response = await fetch(`${this.apiBase}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ arxiv_id: arxivId }),
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }
            
            const result = await response.json();
            this.currentTaskId = result.task_id;
            
            // Show progress section
            this.showProgressSection(arxivId);
            this.startPolling();
            
            this.showToast('Processing started successfully', 'success');
            
        } catch (error) {
            this.showToast(error.message, 'error');
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-magic"></i> Process Paper';
        }
    }
    
    validateArxivId(arxivId) {
        // New format: YYMM.NNNNN or YYMM.NNNNNvN
        const newFormat = /^\d{4}\.\d{4,5}(v\d+)?$/;
        // Old format: archive/YYMMNNN or archive.subj/YYMMNNN
        const oldFormat = /^[a-z\-]+(\.[A-Z]{2})?\/\d{7}$/;
        
        return newFormat.test(arxivId) || oldFormat.test(arxivId);
    }
    
    showProgressSection(arxivId) {
        const progressSection = document.getElementById('progressSection');
        const progressTitle = document.getElementById('progressTitle');
        
        progressTitle.textContent = `Processing arXiv:${arxivId}`;
        progressSection.classList.remove('hidden');
        
        // Reset progress
        this.updateProgress(0, 'Initializing...', '');
    }
    
    hideProgressSection() {
        const progressSection = document.getElementById('progressSection');
        progressSection.classList.add('hidden');
        this.currentTaskId = null;
    }
    
    updateProgress(percentage, stage, details) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressStage = document.getElementById('progressStage');
        const progressDetails = document.getElementById('progressDetails');
        
        if (progressBar) progressBar.style.width = `${percentage}%`;
        if (progressText) progressText.textContent = `${percentage}%`;
        if (progressStage) progressStage.textContent = stage;
        if (progressDetails) progressDetails.textContent = details;
    }
    
    startPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(async () => {
            if (!this.currentTaskId) {
                clearInterval(this.pollingInterval);
                return;
            }
            
            try {
                const response = await fetch(`${this.apiBase}/status/${this.currentTaskId}`);
                const status = await response.json();
                
                this.updateProgress(
                    status.progress,
                    this.getStageText(status.progress),
                    status.error || ''
                );
                
                if (status.status === 'completed') {
                    clearInterval(this.pollingInterval);
                    this.handleProcessingComplete(status.result);
                } else if (status.status === 'failed') {
                    clearInterval(this.pollingInterval);
                    this.handleProcessingFailed(status.error);
                }
                
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 2000);
    }
    
    getStageText(progress) {
        if (progress < 20) return 'Fetching metadata from arXiv...';
        if (progress < 40) return 'Downloading PDF...';
        if (progress < 50) return 'Extracting text from PDF...';
        if (progress < 70) return 'Generating AI summaries...';
        if (progress < 80) return 'Processing sections...';
        if (progress < 90) return 'Extracting keywords...';
        if (progress < 100) return 'Storing results...';
        return 'Complete!';
    }
    
    handleProcessingComplete(result) {
        this.hideProgressSection();
        this.showToast(`Successfully processed: ${result.title}`, 'success');
        document.getElementById('arxivInput').value = '';
        
        // Refresh history and switch to it
        this.loadHistory();
        this.switchTab('history');
    }
    
    handleProcessingFailed(error) {
        this.hideProgressSection();
        this.showToast(`Processing failed: ${error}`, 'error');
    }
    
    async cancelProcessing() {
        if (!this.currentTaskId) return;
        
        try {
            await fetch(`${this.apiBase}/task/${this.currentTaskId}`, {
                method: 'DELETE',
            });
            
            clearInterval(this.pollingInterval);
            this.hideProgressSection();
            this.showToast('Processing cancelled', 'warning');
            
        } catch (error) {
            this.showToast('Failed to cancel processing', 'error');
        }
    }
    
    async loadHistory() {
        const historyContainer = document.getElementById('historyContainer');
        
        try {
            // Show loading state
            historyContainer.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Loading history...</p>
                </div>
            `;
            
            const response = await fetch(`${this.apiBase}/history`);
            const history = await response.json();
            
            this.historyData = history;
            this.filteredHistory = [...history];
            this.renderHistory();
            
        } catch (error) {
            historyContainer.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Failed to load history</p>
                </div>
            `;
            this.showToast('Failed to load history', 'error');
        }
    }
    
    filterHistory(searchTerm) {
        const term = searchTerm.toLowerCase();
        this.filteredHistory = this.historyData.filter(item => 
            item.title.toLowerCase().includes(term) ||
            item.authors.some(author => author.toLowerCase().includes(term)) ||
            item.arxiv_id.toLowerCase().includes(term) ||
            item.keywords.some(keyword => keyword.toLowerCase().includes(term))
        );
        this.renderHistory();
    }
    
    renderHistory() {
        const historyContainer = document.getElementById('historyContainer');
        
        if (this.filteredHistory.length === 0) {
            historyContainer.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-search"></i>
                    <p>No papers found</p>
                </div>
            `;
            return;
        }
        
        const historyHTML = this.filteredHistory.map(item => `
            <div class="history-item" onclick="app.viewDocument('${item.arxiv_id}')">
                <div class="history-item-header">
                    <div>
                        <h3 class="history-item-title">${this.escapeHtml(item.title)}</h3>
                        <div class="history-item-meta">
                            arXiv:${item.arxiv_id} • ${this.formatDate(item.processed_date)}
                        </div>
                    </div>
                </div>
                <div class="history-item-authors">
                    ${item.authors.slice(0, 3).map(author => this.escapeHtml(author)).join(', ')}
                    ${item.authors.length > 3 ? ` and ${item.authors.length - 3} others` : ''}
                </div>
                <div class="history-item-keywords">
                    ${item.keywords.slice(0, 8).map((keyword, index) => 
                        `<span class="keyword-tag ${index >= 5 ? 'secondary' : ''}">${this.escapeHtml(keyword)}</span>`
                    ).join('')}
                    ${item.keywords.length > 8 ? '<span class="keyword-tag secondary">+' + (item.keywords.length - 8) + ' more</span>' : ''}
                </div>
            </div>
        `).join('');
        
        historyContainer.innerHTML = historyHTML;
    }
    
    async viewDocument(arxivId) {
        const modal = document.getElementById('documentModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalContent = document.getElementById('modalContent');
        
        try {
            // Show loading in modal
            modalTitle.textContent = 'Loading...';
            modalContent.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Loading document...</p>
                </div>
            `;
            modal.classList.add('active');
            
            const response = await fetch(`${this.apiBase}/document/${arxivId}`);
            if (!response.ok) {
                throw new Error('Document not found');
            }
            
            const document = await response.json();
            
            modalTitle.textContent = document.title;
            modalContent.innerHTML = this.renderMarkdown(document.content);
            
            // Store current document for export
            this.currentDocument = document;
            
        } catch (error) {
            modalContent.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Failed to load document: ${error.message}</p>
                </div>
            `;
            this.showToast('Failed to load document', 'error');
        }
    }
    
    renderMarkdown(content) {
        // Basic markdown rendering
        let html = content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^(.+)$/gm, '<p>$1</p>')
            .replace(/(<p><br><\/p>)/g, '')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/^<p>(#{1,6})\s+(.+)<\/p>$/gm, (match, hashes, title) => {
                const level = hashes.length;
                return `<h${level}>${title}</h${level}>`;
            });
        
        return html;
    }
    
    closeModal() {
        const modal = document.getElementById('documentModal');
        modal.classList.remove('active');
        this.currentDocument = null;
    }
    
    exportDocument() {
        if (!this.currentDocument) return;
        
        const content = `# ${this.currentDocument.title}\n\n${this.currentDocument.content}`;
        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `arxiv-${this.currentDocument.arxiv_id}.md`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.showToast('Document exported', 'success');
    }
    
    async checkAIStatus() {
        const aiStatus = document.getElementById('aiStatus');
        const aiStatusText = document.getElementById('aiStatusText');
        
        try {
            // Check if we can detect AI capabilities through a health endpoint
            const response = await fetch('/health');
            const health = await response.json();
            
            // For now, assume AI is available if the service is running
            // In a real implementation, you'd check for API key presence
            aiStatus.classList.add('enabled');
            aiStatusText.textContent = 'Enabled';
            
        } catch (error) {
            aiStatus.classList.add('disabled');
            aiStatusText.textContent = 'Disabled';
        }
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
        
        // Click to remove
        toast.addEventListener('click', () => {
            toast.remove();
        });
    }
    
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 1) return '1 day ago';
        if (diffDays < 7) return `${diffDays} days ago`;
        if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
        
        return date.toLocaleDateString();
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ArxivProcessor();
});

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape to close modals
    if (e.key === 'Escape') {
        const modal = document.getElementById('documentModal');
        if (modal && modal.classList.contains('active')) {
            app.closeModal();
        }
    }
    
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const arxivInput = document.getElementById('arxivInput');
        if (arxivInput) {
            arxivInput.focus();
            arxivInput.select();
        }
    }
});
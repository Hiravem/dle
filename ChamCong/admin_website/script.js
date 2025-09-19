// Global variables
let currentPage = 'dashboard';
let employees = [];
let checkins = [];
let notifications = [];
let currentUser = {
    id: 'admin',
    name: 'Admin User',
    role: 'System Administrator'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    loadSampleData();
    setupEventListeners();
    updateDashboard();
    startRealTimeUpdates();
});

// Initialize application
function initializeApp() {
    console.log('Initializing AI Attendance Admin Dashboard...');
    
    // Set up sidebar navigation
    setupSidebarNavigation();
    
    // Set up header interactions
    setupHeaderInteractions();
    
    // Set up form handlers
    setupFormHandlers();
    
    // Initialize charts
    initializeCharts();
    
    console.log('Admin dashboard initialized successfully');
}

// Set up sidebar navigation
function setupSidebarNavigation() {
    const menuItems = document.querySelectorAll('.menu-item');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const sidebar = document.getElementById('sidebar');
    
    menuItems.forEach(item => {
        item.addEventListener('click', function() {
            const page = this.dataset.page;
            navigateToPage(page);
        });
    });
    
    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('collapsed');
    });
    
    mobileMenuBtn.addEventListener('click', function() {
        sidebar.classList.toggle('show');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 1024 && 
            !sidebar.contains(e.target) && 
            !mobileMenuBtn.contains(e.target)) {
            sidebar.classList.remove('show');
        }
    });
}

// Set up header interactions
function setupHeaderInteractions() {
    const notificationBtn = document.getElementById('notificationBtn');
    const notificationDropdown = document.getElementById('notificationDropdown');
    const globalSearch = document.getElementById('globalSearch');
    
    notificationBtn.addEventListener('click', function() {
        notificationDropdown.classList.toggle('show');
        loadNotifications();
    });
    
    globalSearch.addEventListener('input', function() {
        performGlobalSearch(this.value);
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', function(e) {
        if (!notificationBtn.contains(e.target) && !notificationDropdown.contains(e.target)) {
            notificationDropdown.classList.remove('show');
        }
    });
}

// Set up form handlers
function setupFormHandlers() {
    // Employee photo preview
    const photoInput = document.getElementById('empPhoto');
    if (photoInput) {
        photoInput.addEventListener('change', function() {
            previewPhoto(this);
        });
    }
    
    // Range sliders
    setupRangeSliders();
}

// Navigate to page
function navigateToPage(page) {
    // Hide all pages
    const pages = document.querySelectorAll('.page');
    pages.forEach(p => p.style.display = 'none');
    
    // Show selected page
    const selectedPage = document.getElementById(page + 'Page');
    if (selectedPage) {
        selectedPage.style.display = 'block';
    }
    
    // Update active menu item
    const menuItems = document.querySelectorAll('.menu-item');
    menuItems.forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === page) {
            item.classList.add('active');
        }
    });
    
    // Update page title
    const pageTitle = document.getElementById('pageTitle');
    const titles = {
        'dashboard': 'Dashboard',
        'checkins': 'Check-in Records',
        'employees': 'Employee Management',
        'reports': 'Reports & Analytics',
        'analytics': 'AI Analytics',
        'settings': 'System Settings'
    };
    pageTitle.textContent = titles[page] || 'Dashboard';
    
    currentPage = page;
    
    // Load page-specific data
    loadPageData(page);
}

// Load page-specific data
function loadPageData(page) {
    switch(page) {
        case 'dashboard':
            updateDashboard();
            break;
        case 'checkins':
            loadCheckins();
            break;
        case 'employees':
            loadEmployees();
            break;
        case 'reports':
            loadReports();
            break;
        case 'analytics':
            loadAnalytics();
            break;
        case 'settings':
            loadSettings();
            break;
    }
}

// Load sample data
function loadSampleData() {
    // Sample employees
    employees = [
        {
            id: 'EMP001',
            name: 'John Doe',
            department: 'Engineering',
            position: 'Senior Developer',
            photo: 'https://via.placeholder.com/40x40',
            status: 'present',
            lastCheckin: new Date(Date.now() - 2 * 60 * 60 * 1000),
            age: 32,
            salaryLevel: 'senior',
            educationLevel: 'bachelor',
            performanceRating: 4.5
        },
        {
            id: 'EMP002',
            name: 'Jane Smith',
            department: 'Sales',
            position: 'Sales Manager',
            photo: 'https://via.placeholder.com/40x40',
            status: 'late',
            lastCheckin: new Date(Date.now() - 1 * 60 * 60 * 1000),
            age: 28,
            salaryLevel: 'mid',
            educationLevel: 'bachelor',
            performanceRating: 4.2
        },
        {
            id: 'EMP003',
            name: 'Mike Johnson',
            department: 'HR',
            position: 'HR Specialist',
            photo: 'https://via.placeholder.com/40x40',
            status: 'remote',
            lastCheckin: new Date(Date.now() - 30 * 60 * 1000),
            age: 35,
            salaryLevel: 'mid',
            educationLevel: 'master',
            performanceRating: 4.0
        },
        {
            id: 'EMP004',
            name: 'Sarah Wilson',
            department: 'Finance',
            position: 'Financial Analyst',
            photo: 'https://via.placeholder.com/40x40',
            status: 'absent',
            lastCheckin: new Date(Date.now() - 24 * 60 * 60 * 1000),
            age: 29,
            salaryLevel: 'mid',
            educationLevel: 'master',
            performanceRating: 4.3
        }
    ];
    
    // Sample check-ins
    checkins = [
        {
            id: 'CHK001',
            employeeId: 'EMP001',
            employeeName: 'John Doe',
            checkinTime: new Date(Date.now() - 2 * 60 * 60 * 1000),
            location: 'Main Office',
            device: 'camera',
            confidence: 0.95,
            status: 'present',
            isAnomaly: false
        },
        {
            id: 'CHK002',
            employeeId: 'EMP002',
            employeeName: 'Jane Smith',
            checkinTime: new Date(Date.now() - 1 * 60 * 60 * 1000),
            location: 'Main Office',
            device: 'mobile',
            confidence: 0.87,
            status: 'late',
            isAnomaly: false
        },
        {
            id: 'CHK003',
            employeeId: 'EMP003',
            employeeName: 'Mike Johnson',
            checkinTime: new Date(Date.now() - 30 * 60 * 1000),
            location: 'Remote Work Zone',
            device: 'mobile',
            confidence: 0.92,
            status: 'remote',
            isAnomaly: false
        }
    ];
    
    // Sample notifications
    notifications = [
        {
            id: 'NOT001',
            type: 'warning',
            title: 'Anomaly Detected',
            message: 'Unusual check-in pattern detected for EMP004',
            timestamp: new Date(Date.now() - 15 * 60 * 1000),
            read: false
        },
        {
            id: 'NOT002',
            type: 'info',
            title: 'New Employee Registered',
            message: 'Sarah Wilson has been successfully registered',
            timestamp: new Date(Date.now() - 45 * 60 * 1000),
            read: false
        },
        {
            id: 'NOT003',
            type: 'success',
            title: 'System Update',
            message: 'AI models have been successfully retrained',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
            read: true
        }
    ];
}

// Update dashboard
function updateDashboard() {
    updateStats();
    updateRecentCheckins();
    updateEmployeeStatus();
    updateAIInsights();
    updateAttendanceChart();
}

// Update statistics
function updateStats() {
    document.getElementById('totalEmployees').textContent = employees.length;
    document.getElementById('todayCheckins').textContent = checkins.filter(c => 
        isToday(c.checkinTime)
    ).length;
    document.getElementById('anomalies').textContent = checkins.filter(c => c.isAnomaly).length;
    
    const attendanceRate = calculateAttendanceRate();
    document.getElementById('attendanceRate').textContent = attendanceRate + '%';
}

// Update recent check-ins
function updateRecentCheckins() {
    const container = document.getElementById('recentCheckins');
    const recentCheckins = checkins.slice(0, 5);
    
    container.innerHTML = recentCheckins.map(checkin => `
        <div class="activity-item">
            <div class="activity-avatar">
                ${checkin.employeeName.charAt(0)}
            </div>
            <div class="activity-content">
                <div class="activity-title">${checkin.employeeName} checked in</div>
                <div class="activity-subtitle">${checkin.location} • ${checkin.device}</div>
            </div>
            <div class="activity-time">
                ${formatTimeAgo(checkin.checkinTime)}
            </div>
        </div>
    `).join('');
}

// Update employee status
function updateEmployeeStatus() {
    const presentCount = employees.filter(e => e.status === 'present').length;
    const lateCount = employees.filter(e => e.status === 'late').length;
    const absentCount = employees.filter(e => e.status === 'absent').length;
    const remoteCount = employees.filter(e => e.status === 'remote').length;
    
    document.getElementById('presentCount').textContent = presentCount;
    document.getElementById('lateCount').textContent = lateCount;
    document.getElementById('absentCount').textContent = absentCount;
    document.getElementById('remoteCount').textContent = remoteCount;
}

// Update AI insights
function updateAIInsights() {
    const container = document.getElementById('aiInsights');
    
    const insights = [
        {
            icon: 'fas fa-exclamation-triangle',
            title: 'High Turnover Risk',
            description: '3 employees showing signs of potential turnover'
        },
        {
            icon: 'fas fa-chart-line',
            title: 'Attendance Improvement',
            description: 'Attendance rate increased by 8% this month'
        },
        {
            icon: 'fas fa-users',
            title: 'Team Performance',
            description: 'Engineering team has 95% punctuality rate'
        }
    ];
    
    container.innerHTML = insights.map(insight => `
        <div class="insight-item">
            <i class="${insight.icon} insight-icon"></i>
            <div class="insight-content">
                <h4>${insight.title}</h4>
                <p>${insight.description}</p>
            </div>
        </div>
    `).join('');
}

// Load check-ins
function loadCheckins() {
    const tbody = document.getElementById('checkinsTableBody');
    
    tbody.innerHTML = checkins.map(checkin => `
        <tr>
            <td>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div class="activity-avatar">${checkin.employeeName.charAt(0)}</div>
                    <span>${checkin.employeeName}</span>
                </div>
            </td>
            <td>${formatDateTime(checkin.checkinTime)}</td>
            <td>${checkin.location}</td>
            <td>
                <span class="status-badge ${checkin.device === 'camera' ? 'present' : 'remote'}">
                    ${checkin.device}
                </span>
            </td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${checkin.confidence * 100}%"></div>
                </div>
                <span style="font-size: 0.75rem; color: #64748b;">
                    ${Math.round(checkin.confidence * 100)}%
                </span>
            </td>
            <td>
                <span class="status-badge ${checkin.status}">
                    ${checkin.status.charAt(0).toUpperCase() + checkin.status.slice(1)}
                </span>
            </td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewCheckinDetails('${checkin.id}')">
                    <i class="fas fa-eye"></i>
                </button>
                ${checkin.isAnomaly ? '<span class="text-warning"><i class="fas fa-exclamation-triangle"></i></span>' : ''}
            </td>
        </tr>
    `).join('');
}

// Load employees
function loadEmployees() {
    const tbody = document.getElementById('employeesTableBody');
    
    tbody.innerHTML = employees.map(employee => `
        <tr>
            <td>
                <img src="${employee.photo}" alt="${employee.name}" class="employee-photo">
            </td>
            <td>${employee.id}</td>
            <td>${employee.name}</td>
            <td>${employee.department}</td>
            <td>${employee.position}</td>
            <td>
                <span class="status-badge ${employee.status}">
                    ${employee.status.charAt(0).toUpperCase() + employee.status.slice(1)}
                </span>
            </td>
            <td>${employee.lastCheckin ? formatDateTime(employee.lastCheckin) : 'Never'}</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="editEmployee('${employee.id}')">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-danger btn-sm" onclick="deleteEmployee('${employee.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

// Load notifications
function loadNotifications() {
    const container = document.getElementById('notificationList');
    const unreadCount = notifications.filter(n => !n.read).length;
    
    document.getElementById('notificationCount').textContent = unreadCount;
    
    container.innerHTML = notifications.map(notification => `
        <div class="notification-item ${!notification.read ? 'unread' : ''}" 
             onclick="markNotificationAsRead('${notification.id}')">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <i class="fas fa-${getNotificationIcon(notification.type)}" 
                   style="color: ${getNotificationColor(notification.type)}; margin-top: 0.125rem;"></i>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">
                        ${notification.title}
                    </div>
                    <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">
                        ${notification.message}
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">
                        ${formatTimeAgo(notification.timestamp)}
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Initialize charts
function initializeCharts() {
    // This would integrate with Chart.js or similar library
    console.log('Charts initialized');
}

// Set up range sliders
function setupRangeSliders() {
    const sliders = [
        { id: 'faceThreshold', displayId: 'faceThresholdValue' },
        { id: 'livenessSensitivity', displayId: 'livenessSensitivityValue' },
        { id: 'anomalySensitivity', displayId: 'anomalySensitivityValue' }
    ];
    
    sliders.forEach(slider => {
        const sliderElement = document.getElementById(slider.id);
        const displayElement = document.getElementById(slider.displayId);
        
        if (sliderElement && displayElement) {
            sliderElement.addEventListener('input', function() {
                displayElement.textContent = this.value;
            });
        }
    });
}

// Employee management functions
function openAddEmployeeModal() {
    const modal = document.getElementById('addEmployeeModal');
    modal.classList.add('show');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('show');
}

function previewPhoto(input) {
    const preview = document.getElementById('photoPreview');
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function submitAddEmployee() {
    const form = document.getElementById('addEmployeeForm');
    const formData = new FormData(form);
    
    // Simulate API call
    showLoading();
    
    setTimeout(() => {
        const newEmployee = {
            id: formData.get('employee_id'),
            name: formData.get('name'),
            department: formData.get('department'),
            position: formData.get('position'),
            photo: URL.createObjectURL(formData.get('face_image')),
            status: 'absent',
            lastCheckin: null,
            age: parseInt(formData.get('age')),
            salaryLevel: formData.get('salary_level'),
            educationLevel: formData.get('education_level'),
            performanceRating: 4.0
        };
        
        employees.push(newEmployee);
        
        hideLoading();
        closeModal('addEmployeeModal');
        showToast('Employee added successfully', 'success');
        
        if (currentPage === 'employees') {
            loadEmployees();
        }
        updateDashboard();
    }, 2000);
}

function editEmployee(employeeId) {
    const employee = employees.find(e => e.id === employeeId);
    if (employee) {
        // Populate edit form
        document.getElementById('editEmpId').value = employee.id;
        // ... populate other fields
        
        const modal = document.getElementById('editEmployeeModal');
        modal.classList.add('show');
    }
}

function deleteEmployee(employeeId) {
    if (confirm('Are you sure you want to delete this employee?')) {
        const index = employees.findIndex(e => e.id === employeeId);
        if (index > -1) {
            employees.splice(index, 1);
            showToast('Employee deleted successfully', 'success');
            loadEmployees();
            updateDashboard();
        }
    }
}

function submitEditEmployee() {
    // Similar to submitAddEmployee but for editing
    showToast('Employee updated successfully', 'success');
    closeModal('editEmployeeModal');
    loadEmployees();
}

// Utility functions
function formatDateTime(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function formatTimeAgo(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
}

function isToday(date) {
    const today = new Date();
    return date.toDateString() === today.toDateString();
}

function calculateAttendanceRate() {
    const todayCheckins = checkins.filter(c => isToday(c.checkinTime));
    const totalEmployees = employees.length;
    return Math.round((todayCheckins.length / totalEmployees) * 100);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function getNotificationColor(type) {
    const colors = {
        'success': '#10b981',
        'error': '#ef4444',
        'warning': '#f59e0b',
        'info': '#3b82f6'
    };
    return colors[type] || '#3b82f6';
}

function markNotificationAsRead(notificationId) {
    const notification = notifications.find(n => n.id === notificationId);
    if (notification) {
        notification.read = true;
        loadNotifications();
    }
}

function markAllAsRead() {
    notifications.forEach(n => n.read = true);
    loadNotifications();
}

function performGlobalSearch(query) {
    if (query.length < 2) return;
    
    const results = {
        employees: employees.filter(e => 
            e.name.toLowerCase().includes(query.toLowerCase()) ||
            e.id.toLowerCase().includes(query.toLowerCase())
        ),
        checkins: checkins.filter(c =>
            c.employeeName.toLowerCase().includes(query.toLowerCase())
        )
    };
    
    console.log('Search results:', results);
    // Implement search results display
}

function toggleUserMenu() {
    const dropdown = document.getElementById('userMenuDropdown');
    dropdown.classList.toggle('show');
}

function logout() {
    if (confirm('Are you sure you want to logout?')) {
        showToast('Logged out successfully', 'info');
        // Implement logout logic
    }
}

function showLoading() {
    document.getElementById('loadingOverlay').classList.add('show');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = getNotificationIcon(type);
    const color = getNotificationColor(type);
    
    toast.innerHTML = `
        <i class="fas fa-${icon}" style="color: ${color};"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; margin-left: auto; cursor: pointer;">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Start real-time updates
function startRealTimeUpdates() {
    // Simulate real-time updates
    setInterval(() => {
        if (Math.random() < 0.1) { // 10% chance every 5 seconds
            addRandomCheckin();
        }
    }, 5000);
}

function addRandomCheckin() {
    const employee = employees[Math.floor(Math.random() * employees.length)];
    const newCheckin = {
        id: 'CHK' + Date.now(),
        employeeId: employee.id,
        employeeName: employee.name,
        checkinTime: new Date(),
        location: 'Main Office',
        device: Math.random() > 0.5 ? 'camera' : 'mobile',
        confidence: 0.8 + Math.random() * 0.2,
        status: 'present',
        isAnomaly: Math.random() < 0.05
    };
    
    checkins.unshift(newCheckin);
    
    if (currentPage === 'dashboard') {
        updateDashboard();
    } else if (currentPage === 'checkins') {
        loadCheckins();
    }
    
    showToast(`${employee.name} just checked in`, 'info');
}

// Event listeners setup
function setupEventListeners() {
    // Close modals when clicking outside
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal')) {
            e.target.classList.remove('show');
        }
    });
    
    // Handle window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 1024) {
            document.getElementById('sidebar').classList.remove('show');
        }
    });
}

// Placeholder functions for future implementation
function viewAllCheckins() {
    navigateToPage('checkins');
}

function viewCheckinDetails(checkinId) {
    showToast('Check-in details feature coming soon', 'info');
}

function exportCheckins() {
    showToast('Exporting check-ins...', 'info');
}

function exportReport() {
    showToast('Exporting report...', 'info');
}

function generateReport() {
    showToast('Generating report...', 'info');
}

function loadReports() {
    showToast('Loading reports...', 'info');
}

function loadAnalytics() {
    showToast('Loading analytics...', 'info');
}

function loadSettings() {
    showToast('Loading settings...', 'info');
}

function openProfile() {
    showToast('Profile page coming soon', 'info');
}

function openSettings() {
    navigateToPage('settings');
}

function openAddGeofenceModal() {
    showToast('Add geofence feature coming soon', 'info');
}

// API integration functions (placeholder)
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const response = await fetch(`/api/${endpoint}`, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: data ? JSON.stringify(data) : null
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showToast('API call failed: ' + error.message, 'error');
        throw error;
    }
}

// Initialize everything when DOM is loaded
console.log('AI Attendance Admin Dashboard script loaded');

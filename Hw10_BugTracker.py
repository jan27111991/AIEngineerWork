class BugTracker:
    def __init__(self):
        """Initialize the BugTracker with an empty dictionary for bugs"""
        self.bugs = {}
    
    def add_bug(self, bug_id, description, severity):
        """
        Add a new bug to the tracker
        
        Args:
            bug_id (str/int): Unique identifier for the bug
            description (str): Description of the bug
            severity (str): Severity level (e.g., Low, Medium, High, Critical)
        """
        if bug_id in self.bugs:
            print(f"Bug ID {bug_id} already exists!")
            return False
        
        self.bugs[bug_id] = {
            'description': description,
            'severity': severity,
            'status': 'Open'
        }
        print(f"Bug {bug_id} added successfully!")
        return True
    
    def update_status(self, bug_id, new_status):
        """
        Update the status of an existing bug
        
        Args:
            bug_id (str/int): ID of the bug to update
            new_status (str): New status (e.g., Open, In Progress, Closed)
        """
        if bug_id not in self.bugs:
            print(f"Bug ID {bug_id} not found!")
            return False
        
        self.bugs[bug_id]['status'] = new_status
        print(f"Bug {bug_id} status updated to '{new_status}'")
        return True
    
    def list_all_bugs(self):
        """Display all bugs in a readable format"""
        if not self.bugs:
            print("No bugs found in the tracker!")
            return
        
        print("\n" + "="*60)
        print("BUG TRACKER - ALL BUGS")
        print("="*60)
        
        for bug_id, details in self.bugs.items():
            print(f"Bug ID: {bug_id}")
            print(f"Description: {details['description']}")
            print(f"Severity: {details['severity']}")
            print(f"Status: {details['status']}")
            print("-" * 40)
    
    def get_bug_count_by_status(self):
        """Bonus method: Get count of bugs by status"""
        status_count = {}
        for bug in self.bugs.values():
            status = bug['status']
            status_count[status] = status_count.get(status, 0) + 1
        return status_count


if __name__ == "__main__":
    # Create a BugTracker object
    tracker = BugTracker()
    
    # Add three bugs with different details
    print("Adding bugs to the tracker...")
    tracker.add_bug("BUG-001", "Login button not working", "High")
    tracker.add_bug("BUG-002", "Typo in welcome message", "Low")
    tracker.add_bug("BUG-003", "Database connection timeout", "Critical")
    
    # Update status of bugs
    print("\nUpdating bug statuses...")
    tracker.update_status("BUG-001", "In Progress")
    tracker.update_status("BUG-003", "Closed")
    
    # Display all bugs
    tracker.list_all_bugs()
    
    # Bonus: Show bug count by status
    status_counts = tracker.get_bug_count_by_status()
    print(f"\nBug Count by Status: {status_counts}")
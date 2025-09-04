function triggerAlarm() {
    const statusElement = document.getElementById('status');
    statusElement.textContent = 'Fire Detected! Alarm Activated!';
    alert('Alarm has been triggered! Notifications will be sent.');
}
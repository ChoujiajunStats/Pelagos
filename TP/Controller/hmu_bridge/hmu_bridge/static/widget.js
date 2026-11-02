"use strict";

const $ = (id) => document.getElementById(id);
const wsScheme = location.protocol === "https:" ? "wss:" : "ws:";
const controlUrl = `${wsScheme}//${location.host}/ws/control`;
const variables = [
  "inputs/mavlink/axis-x",
  "inputs/mavlink/axis-y",
  "inputs/mavlink/axis-z",
  "inputs/mavlink/axis-r",
];

let socket = null;
let token = null;
let sequence = 0;
let state = null;
let reconnectTimer = null;
let armTimer = null;
let lastError = "";
const rawAxes = { x: 0, y: 0, z: 500, r: 0 };
const controls = {
  light: 0,
  gear: 1,
  heading_hold: false,
  depth_hold: false,
  gripper: 0,
  gimbal_1: 0,
  gimbal_2: 0,
};

function clamp(value, low = -1, high = 1) {
  return Math.max(low, Math.min(high, Number(value) || 0));
}

function deadband(value) {
  return Math.abs(value) < 0.06 ? 0 : value;
}

function inverted(name) {
  return document.querySelector(`[data-invert="${name}"]`).checked ? -1 : 1;
}

function axes() {
  return {
    surge: deadband(clamp(rawAxes.x / 1000)) * inverted("surge"),
    sway: deadband(clamp(rawAxes.y / 1000)) * inverted("sway"),
    heave: deadband(clamp((rawAxes.z - 500) / 500)) * inverted("heave"),
    yaw: deadband(clamp(rawAxes.r / 1000)) * inverted("yaw"),
  };
}

function send(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

function sendControl() {
  if (!token || document.hidden) return;
  send({ type: "control", token, sequence: ++sequence, axes: axes(), ...controls });
  renderAxes();
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(controlUrl);
  socket.addEventListener("open", () => {
    lastError = "";
    render();
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "state") state = message.state;
    if (message.type === "takeover") {
      token = message.token;
      sequence = 0;
    }
    if (message.type === "error") lastError = message.message;
    render();
  });
  socket.addEventListener("close", () => {
    token = null;
    socket = null;
    render();
    reconnectTimer = setTimeout(connect, 1200);
  });
  socket.addEventListener("error", () => socket.close());
}

function text(id, value, digits = 0) {
  $(id).textContent = value === null || value === undefined ? "--" : Number(value).toFixed(digits);
}

function render() {
  const rov = state?.rov || {};
  const main = state?.main || {};
  const safety = state?.control || {};
  const mode = rov.mode || "observe";
  const motorProtocol = rov.command_protocol === "motor";
  $("modeBadge").textContent = mode === "control" ? "控制" : "只读";
  $("modeBadge").className = `badge ${mode}`;
  $("endpoint").textContent = `${rov.host || "192.168.1.7"}:${rov.port || 8887}`;
  $("linkBadge").className = `link-status ${rov.connected ? "online" : "offline"}`;
  $("linkBadge").querySelector("span").textContent = rov.connected ? "ROV 在线" : "ROV 离线";

  text("depth", main.depth_m, 2);
  text("heading", main.heading_deg);
  text("pitch", main.pitch_deg);
  text("roll", main.roll_deg);
  text("battery", main.battery_percent);
  text("temperature", main.water_temperature_c);

  const canTakeover = mode === "control" && rov.connected && !token;
  $("takeover").disabled = !canTakeover;
  $("arm").disabled = !token || safety.armed;
  $("disarm").disabled = !token || !safety.armed;
  $("release").disabled = !token;
  $("headingHold").disabled = motorProtocol;
  $("depthHold").disabled = motorProtocol;
  for (const button of $("gear").querySelectorAll("button")) button.disabled = motorProtocol;
  if (motorProtocol) {
    controls.heading_hold = false;
    controls.depth_hold = false;
  }
  $("controlState").textContent = safety.armed ? "ARMED / 控制输出" : token ? "已接管 / DISARMED" : "未接管";
  $("lastFrame").textContent = state?.last_frame_at ? `遥测 ${new Date(state.last_frame_at).toLocaleTimeString()}` : "等待遥测";

  const alert = $("alert");
  const leak = main.leak_detected === true;
  const message = leak ? "检测到漏水信号" : lastError || (!rov.connected ? rov.last_error || "ROV 连接断开" : "");
  alert.textContent = message;
  alert.classList.toggle("hidden", !message);
  alert.classList.toggle("warning", Boolean(lastError) && !leak);
}

function renderAxes() {
  const current = axes();
  for (const [name, value] of Object.entries(current)) {
    $(`${name}Value`).textContent = value.toFixed(2);
    const bar = $(`${name}Bar`);
    const width = Math.abs(value) * 50;
    bar.style.width = `${width}%`;
    bar.style.left = value < 0 ? `${50 - width}%` : "50%";
  }
}

window.addEventListener("message", (event) => {
  if (event.data?.type !== "cockpit:datalakeVariable") return;
  const key = event.data.variable.split("-").at(-1);
  if (key in rawAxes) rawAxes[key] = Number(event.data.value) || 0;
});

for (const variable of variables) {
  window.parent.postMessage({ type: "cockpit:listenToDatalakeVariables", variable, maxRateHz: 25 }, "*");
}

$("takeover").addEventListener("click", () => send({ type: "takeover" }));
$("disarm").addEventListener("click", () => token && send({ type: "disarm", token }));
$("release").addEventListener("click", () => {
  if (token) send({ type: "release", token });
  token = null;
  render();
});

function startArm(event) {
  event.preventDefault();
  if (!token || armTimer) return;
  lastError = "";
  sendControl();
  send({ type: "arm_begin", token });
  $("arm").classList.add("holding");
  armTimer = setTimeout(() => {
    send({ type: "arm_commit", token });
    armTimer = null;
    $("arm").classList.remove("holding");
  }, 1550);
}

function cancelArm() {
  if (!armTimer) return;
  clearTimeout(armTimer);
  armTimer = null;
  $("arm").classList.remove("holding");
  if (token) send({ type: "arm_cancel", token });
}

$("arm").addEventListener("pointerdown", startArm);
$("arm").addEventListener("pointerup", cancelArm);
$("arm").addEventListener("pointerleave", cancelArm);
$("arm").addEventListener("pointercancel", cancelArm);

$("light").addEventListener("input", (event) => {
  controls.light = Number(event.target.value);
  $("lightValue").textContent = `${controls.light}%`;
});
$("gear").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  controls.gear = Number(button.dataset.value);
  for (const item of $("gear").querySelectorAll("button")) item.classList.toggle("selected", item === button);
});
$("headingHold").addEventListener("change", (event) => controls.heading_hold = event.target.checked);
$("depthHold").addEventListener("change", (event) => controls.depth_hold = event.target.checked);
$("gripper").addEventListener("change", (event) => controls.gripper = Number(event.target.checked));

for (const button of document.querySelectorAll("[data-gimbal]")) {
  const field = `gimbal_${button.dataset.gimbal}`;
  const start = (event) => {
    event.preventDefault();
    controls[field] = button.dataset.direction === "up" ? 0x01 : 0x11;
  };
  const stop = () => controls[field] = 0;
  button.addEventListener("pointerdown", start);
  button.addEventListener("pointerup", stop);
  button.addEventListener("pointerleave", stop);
  button.addEventListener("pointercancel", stop);
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden && token) send({ type: "disarm", token });
});
window.addEventListener("beforeunload", () => {
  if (token) send({ type: "release", token });
});

for (const checkbox of document.querySelectorAll("[data-invert]")) {
  const key = `hmu-invert-${checkbox.dataset.invert}`;
  checkbox.checked = localStorage.getItem(key) === "true";
  checkbox.addEventListener("change", () => localStorage.setItem(key, String(checkbox.checked)));
}

setInterval(sendControl, 50);
setInterval(render, 250);
renderAxes();
render();
connect();

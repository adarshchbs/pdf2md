import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

const target = document.getElementById('app');
if (!target) throw new Error('Dashboard root element is unavailable.');
mount(App, { target });

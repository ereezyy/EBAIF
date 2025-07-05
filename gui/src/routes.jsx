import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

import Dashboard from './pages/Dashboard'
import Agents from './pages/Agents'
import Tasks from './pages/Tasks'
import Communications from './pages/Communications'
import Settings from './pages/Settings'
import Analytics from './pages/Analytics'
import MainLayout from './layouts/MainLayout'

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="agents" element={<Agents />} />
        <Route path="tasks" element={<Tasks />} />
        <Route path="communications" element={<Communications />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}

export default AppRoutes
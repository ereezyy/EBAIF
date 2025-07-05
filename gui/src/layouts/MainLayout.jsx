import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Box, useMediaQuery, useTheme } from '@mui/material'
import Sidebar from '../components/Sidebar'
import Topbar from '../components/Topbar'

const MainLayout = () => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)

  const handleToggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar open={sidebarOpen} onToggle={handleToggleSidebar} />
      <Box sx={{ 
        flexGrow: 1,
        ml: sidebarOpen ? { xs: 0, sm: '80px', md: '280px' } : '80px',
        transition: 'margin 0.3s ease'
      }}>
        <Topbar />
        <Box component="main" sx={{ flexGrow: 1, p: { xs: 1, sm: 2, md: 3 } }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
}

export default MainLayout
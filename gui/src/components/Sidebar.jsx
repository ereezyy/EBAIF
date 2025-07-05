import React from 'react'
import { 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  IconButton,
  Box,
  Typography,
  Chip,
  Divider
} from '@mui/material'
import { useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Dashboard,
  SmartToy,
  Assignment,
  Message,
  Analytics,
  Settings,
  Menu,
  Computer,
  Zap
} from '@mui/icons-material'

const menuItems = [
  { text: 'Dashboard', icon: Dashboard, path: '/dashboard' },
  { text: 'Agents', icon: SmartToy, path: '/agents' },
  { text: 'Tasks', icon: Assignment, path: '/tasks' },
  { text: 'Communications', icon: Message, path: '/communications' },
  { text: 'Analytics', icon: Analytics, path: '/analytics' },
  { text: 'Settings', icon: Settings, path: '/settings' }
]

const Sidebar = ({ open, onToggle }) => {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: open ? 280 : 80,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? 280 : 80,
          boxSizing: 'border-box',
          background: 'rgba(10, 14, 26, 0.95)',
          backdropFilter: 'blur(20px)',
          border: 'none',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
          transition: 'width 0.3s ease',
          overflow: 'hidden'
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <IconButton onClick={onToggle} sx={{ color: 'white' }}>
            <Menu />
          </IconButton>
          {open && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Box sx={{ ml: 1, display: 'flex', alignItems: 'center' }}>
                <Computer sx={{ color: '#667eea', mr: 1 }} />
                <Typography variant="h6" sx={{ color: 'white', fontWeight: 700 }}>
                  EBAIF
                </Typography>
              </Box>
            </motion.div>
          )}
        </Box>

        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Box sx={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              p: 2,
              mb: 3,
              textAlign: 'center'
            }}>
              <Zap sx={{ color: 'white', fontSize: 40, mb: 1 }} />
              <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
                Autonomous AI System
              </Typography>
              <Chip 
                label="ACTIVE" 
                size="small" 
                sx={{ 
                  mt: 1,
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  fontWeight: 600
                }} 
              />
            </Box>
          </motion.div>
        )}

        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', mb: 2 }} />

        <List>
          {menuItems.map((item, index) => (
            <ListItem
              key={item.path}
              button
              onClick={() => navigate(item.path)}
              sx={{
                borderRadius: '12px',
                mb: 1,
                background: location.pathname === item.path 
                  ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%)'
                  : 'transparent',
                border: location.pathname === item.path 
                  ? '1px solid rgba(102, 126, 234, 0.5)'
                  : '1px solid transparent',
                transition: 'all 0.3s ease',
                '&:hover': {
                  background: 'rgba(102, 126, 234, 0.1)',
                  border: '1px solid rgba(102, 126, 234, 0.3)'
                }
              }}
            >
              <ListItemIcon sx={{ 
                color: location.pathname === item.path ? '#667eea' : '#b8c5d6',
                minWidth: open ? 56 : 40
              }}>
                <item.icon />
              </ListItemIcon>
              {open && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <ListItemText 
                    primary={item.text}
                    sx={{
                      '& .MuiListItemText-primary': {
                        color: location.pathname === item.path ? 'white' : '#b8c5d6',
                        fontWeight: location.pathname === item.path ? 600 : 400
                      }
                    }}
                  />
                </motion.div>
              )}
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}

export default Sidebar